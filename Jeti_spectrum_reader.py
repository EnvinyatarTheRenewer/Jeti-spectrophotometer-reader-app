"""
Spectrum Plotter
=================

A small Tkinter application for exploring JETI spectrophotometer CSV
exports.

Workflow
--------
1. Pick a folder containing spectrum CSV files.
2. Build one or more "plot groups": each group is a set of observations
   (CSV columns) that will be drawn together. Every group gets its own
   title, plus a custom legend label and colour per observation.
3. Choose a wavelength window shared by all groups.
4. Choose whether to export raw radiance plots, AUC-normalized plots,
   or both.

Notes on the input format
--------------------------
Historically, JETI export files split every numeric radiance value
across two CSV fields (e.g. ``1234,56`` instead of ``1234.56``), which
is what you get when a spreadsheet tool re-saves a CSV under a locale
that uses ``,`` as the decimal separator. We reconstruct
``<int_part>.<frac_part>`` ourselves rather than relying on ``locale``
or pandas' decimal handling.

Because JETI may fix this on their end at some point, `read_spectrum_csv`
first *detects* which layout a given file actually uses (one field per
value vs. two split fields per value) instead of assuming the historical
bug is still present. See `_detect_value_field_layout`.
"""

from __future__ import annotations

import glob
import logging
import os
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("spectrum_plotter")


# =====================================================================
# Configuration
# =====================================================================

APP_TITLE = "Spectrum Plotter"

RAW_FOLDER_NAME = "Spectrum_plots_RAW"
AUC_FOLDER_NAME = "Spectrum_plots_AUC"

BUTTON_COLUMNS = 3

EXCLUDED_KEYWORDS = ("fresh", "initcond")

COLOR_PALETTE = [
    "#18324A", "#28566B", "#3F7882", "#61938E", "#82A99A",
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]


# =====================================================================
# Data model
# =====================================================================

@dataclass
class Spectrum:
    """A single observation (one CSV column) with its radiance data."""

    name: str              # cleaned/display-friendly name
    original_name: str     # exact column header from the CSV
    source_file: str       # source filename, for error messages
    wavelength_nm: np.ndarray
    radiance: np.ndarray

    def duplicate_key(self) -> tuple:
        """Key used to detect two `Spectrum`s holding identical data."""
        return (
            tuple(np.round(self.wavelength_nm, 12)),
            tuple(np.round(self.radiance, 12)),
        )


@dataclass
class GroupItem:
    """One row of a plot group: which column, what label/colour to use."""

    column_name: str
    display_name: str
    color: str


@dataclass
class PlotGroup:
    """A named set of observations that will be drawn on the same figure."""

    title: str
    items: list[GroupItem] = field(default_factory=list)


# =====================================================================
# CSV parsing
# =====================================================================

def read_spectrum_csv(filepath: Path) -> pd.DataFrame:
    """Parse one JETI export into a DataFrame of wavelength + spectra.

    The file has a ``Name,<spec1>,<spec2>,...`` header line identifying
    each spectrum, followed later by a radiance block starting with
    ``Wavelength [nm],Le [W/(sr*sqm*nm)],...``.
    """
    with open(filepath, "r", encoding="latin1") as f:
        all_lines = f.readlines()

    name_line_index = next(
        (i for i, line in enumerate(all_lines) if line.startswith("Name,")), None
    )
    if name_line_index is None:
        raise ValueError("Could not find the 'Name' line.")

    spectrum_names = all_lines[name_line_index].strip().split(",")[1:]
    spectrum_count = len(spectrum_names)

    radiance_header_index = next(
        (
            i for i, line in enumerate(all_lines)
            if line.startswith("Wavelength [nm],") and "Le [W/(sr*sqm*nm)]" in line
        ),
        None,
    )
    if radiance_header_index is None:
        raise ValueError("Could not find the radiance spectrum block.")

    radiance_lines = all_lines[radiance_header_index + 1:]
    data_rows = _parse_radiance_block(radiance_lines, spectrum_count, filename=os.fspath(filepath))
    if not data_rows:
        raise ValueError("No spectral data found.")

    column_names = ["wavelength"] + _make_unique_names(spectrum_names)
    spectra_df = pd.DataFrame(data_rows, columns=column_names)
    spectra_df = spectra_df.drop_duplicates(subset="wavelength", keep="first")
    spectra_df = spectra_df.sort_values("wavelength").reset_index(drop=True)
    return spectra_df


def _detect_value_field_layout(
    radiance_lines: list[str], spectrum_count: int
) -> tuple[int, bool]:
    """Work out how radiance values are encoded in this particular file.

    Historically JETI split every decimal value across two CSV fields
    (integer part, fractional part), e.g. ``1234,56`` for ``1234.56``.
    If that bug is ever fixed upstream, each value will instead sit in
    a single field with a real decimal point, e.g. ``1234.56``.

    We tell the two layouts apart by counting how many comma-separated
    fields the first real data row actually has:
      * ``1 (wavelength) + 2 * spectrum_count`` fields  -> split/legacy layout
      * ``1 (wavelength) + 1 * spectrum_count`` fields   -> fixed/combined layout

    Returns
    -------
    (expected_field_count, uses_split_decimal_fields)
    """
    split_layout_field_count = 1 + 2 * spectrum_count
    combined_layout_field_count = 1 + spectrum_count

    for raw_line in radiance_lines:
        line = raw_line.strip()
        if not line or line.startswith("Wavelength [nm]"):
            break

        field_count = len(line.split(","))

        if field_count >= split_layout_field_count:
            return split_layout_field_count, True
        if field_count >= combined_layout_field_count:
            return combined_layout_field_count, False
    return split_layout_field_count, True


def _parse_radiance_block(
    radiance_lines: list[str], spectrum_count: int, filename: str = ""
) -> list[list[float]]:
    """Turn the raw radiance lines into rows of [wavelength, *radiance_values]."""
    expected_field_count, uses_split_decimal_fields = _detect_value_field_layout(
        radiance_lines, spectrum_count
    )

    if uses_split_decimal_fields:
        log.debug(f"{filename}: detected legacy split-decimal CSV layout.")
    else:
        log.debug(f"{filename}: detected fixed single-field decimal CSV layout.")

    parsed_rows: list[list[float]] = []

    for raw_line in radiance_lines:
        line = raw_line.strip()
        if not line or line.startswith("Wavelength [nm]"):
            break

        fields = line.split(",")
        if len(fields) < expected_field_count:
            continue

        try:
            wavelength_nm = float(fields[0])
        except ValueError:
            continue

        if uses_split_decimal_fields:
            radiance_values, all_valid = _reconstruct_split_decimal_values(fields, spectrum_count)
        else:
            radiance_values, all_valid = _read_combined_decimal_values(fields, spectrum_count)

        if all_valid:
            parsed_rows.append([wavelength_nm] + radiance_values)

    return parsed_rows


def _reconstruct_split_decimal_values(
    fields: list[str], spectrum_count: int
) -> tuple[list[float], bool]:
    """Legacy layout: glue ``int_part`` + ``frac_part`` field pairs back
    into floats, e.g. fields ["1234", "56"] -> 1234.56."""
    radiance_values: list[float] = []
    for spectrum_index in range(spectrum_count):
        integer_part = fields[1 + 2 * spectrum_index]
        fractional_part = fields[2 + 2 * spectrum_index]
        try:
            radiance_values.append(float(f"{integer_part}.{fractional_part}"))
        except ValueError:
            return radiance_values, False
    return radiance_values, True


def _read_combined_decimal_values(
    fields: list[str], spectrum_count: int
) -> tuple[list[float], bool]:
    """Fixed layout: each value is already a single properly formatted
    field. Small safety net: if a value still uses a comma decimal
    separator on its own (e.g. "1234,56" as ONE field), fall back to
    treating the comma as a decimal point rather than failing outright."""
    radiance_values: list[float] = []
    for spectrum_index in range(spectrum_count):
        raw_value = fields[1 + spectrum_index]
        try:
            radiance_values.append(float(raw_value))
        except ValueError:
            try:
                radiance_values.append(float(raw_value.replace(",", ".")))
            except ValueError:
                return radiance_values, False
    return radiance_values, True


def _make_unique_names(names: list[str]) -> list[str]:
    """Disambiguate repeated column headers (e.g. two 'sample_1' columns)."""
    seen_counts: dict[str, int] = {}
    unique_names: list[str] = []
    for name in names:
        if name not in seen_counts:
            seen_counts[name] = 0
            unique_names.append(name)
        else:
            seen_counts[name] += 1
            unique_names.append(f"{name}__duplicate{seen_counts[name]}")
    return unique_names


def clean_spectrum_name(name: str) -> str:
    """Strip the internal '__duplicateN' suffix for display purposes."""
    return name.split("__duplicate")[0] if "__duplicate" in name else name


def is_excluded_spectrum(name: str) -> bool:
    """Fresh-leaf and pre-burial ('initcond') samples are never plotted."""
    lower_name = name.lower()
    return any(keyword in lower_name for keyword in EXCLUDED_KEYWORDS)


# =====================================================================
# Directory-level loading
# =====================================================================

def load_all_spectra(data_dir: Path) -> tuple[list[Spectrum], list[str]]:
    """Load every CSV in `data_dir` into a flat list of `Spectrum` objects.

    Returns (spectra, warnings). Warnings are collected rather than
    raised so that one bad file doesn't block the rest of the folder.
    """
    csv_filepaths = sorted(glob.glob(str(Path(data_dir) / "*.csv")))
    if not csv_filepaths:
        raise FileNotFoundError("No CSV files were found in the selected directory.")

    all_spectra: list[Spectrum] = []
    warnings: list[str] = []

    for filepath in csv_filepaths:
        filename = os.path.basename(filepath)
        try:
            spectra_df = read_spectrum_csv(Path(filepath))
        except Exception as exc:
            warnings.append(f"{filename}: {exc}")
            continue

        all_spectra.extend(_spectra_from_dataframe(spectra_df, filename, warnings))

    if not all_spectra:
        raise RuntimeError("No valid spectra were found.")

    return all_spectra, warnings


def _spectra_from_dataframe(
    spectra_df: pd.DataFrame, filename: str, warnings: list[str]
) -> list[Spectrum]:
    """Extract one `Spectrum` per data column, skipping excluded/invalid ones.

    Columns are read by position (not by name) since duplicate headers
    are common in these exports.
    """
    wavelength_nm = spectra_df.iloc[:, 0].to_numpy(dtype=float)
    extracted_spectra: list[Spectrum] = []

    for column_index in range(1, len(spectra_df.columns)):
        raw_column_name = spectra_df.columns[column_index]
        if is_excluded_spectrum(raw_column_name):
            continue

        radiance = spectra_df.iloc[:, column_index].to_numpy(dtype=float)
        valid_mask = np.isfinite(wavelength_nm) & np.isfinite(radiance)
        valid_wavelength = wavelength_nm[valid_mask]
        valid_radiance = radiance[valid_mask]

        if len(valid_radiance) < 2:
            warnings.append(f"{filename}: Not enough valid data for '{raw_column_name}'")
            continue

        extracted_spectra.append(
            Spectrum(
                name=clean_spectrum_name(raw_column_name),
                original_name=raw_column_name,
                source_file=filename,
                wavelength_nm=valid_wavelength,
                radiance=valid_radiance,
            )
        )

    return extracted_spectra


def get_unique_column_names(spectra: list[Spectrum]) -> list[str]:
    """All distinct display names, sorted case-insensitively."""
    return sorted({spectrum.name for spectrum in spectra}, key=str.lower)


def remove_identical_spectra(spectra: list[Spectrum]) -> list[Spectrum]:
    """Drop spectra that are exact data duplicates of an earlier one."""
    seen_keys: set[tuple] = set()
    unique_spectra: list[Spectrum] = []
    for spectrum in spectra:
        key = spectrum.duplicate_key()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_spectra.append(spectrum)
    return unique_spectra


def slugify_for_filename(text: str) -> str:
    """Turn an arbitrary plot title into a filesystem-safe filename fragment."""
    safe_chars = [c if (c.isalnum() or c in ("-", "_")) else "_" for c in text.strip()]
    slug = "".join(safe_chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "plot"


# =====================================================================
# GUI application
# =====================================================================

class SpectrumPlotterApp:
    """Wizard-style Tkinter UI driving the load -> group -> plot flow."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1000x750")
        self.root.minsize(800, 600)

        # Data loaded from disk
        self.data_dir: Path | None = None
        self.all_spectra: list[Spectrum] = []
        self.column_names: list[str] = []

        # Wizard state carried between screens
        self.plot_groups: list[PlotGroup] = []
        self.wavelength_min_nm: float = 0.0
        self.wavelength_max_nm: float = 0.0
        self.plot_type: tk.StringVar

        # Transient per-screen widget state
        self.selected_column_names: set[str] = set()
        self.column_buttons_by_name: dict[str, tk.Button] = {}
        self.group_title_entry: ttk.Entry
        self.customization_name_entries: list[ttk.Entry] = []
        self.customization_color_entries: list[dict] = []

        self.show_directory_screen()

    # -----------------------------------------------------------------
    # Shared helpers
    # -----------------------------------------------------------------

    def clear_window(self) -> None:
        for widget in self.root.winfo_children():
            widget.destroy()

    def _make_scrollable_frame(self, parent: tk.Widget) -> ttk.Frame:
        """Return a scrollable content frame packed into `parent`."""
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        content_frame = ttk.Frame(canvas)

        content_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return content_frame

    # -----------------------------------------------------------------
    # Screen 1: choose directory
    # -----------------------------------------------------------------

    def show_directory_screen(self) -> None:
        self.clear_window()
        frame = ttk.Frame(self.root, padding=30)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Spectrum Plotter", font=("TkDefaultFont", 20, "bold")).pack(pady=(20, 10))
        ttk.Label(
            frame, text="Enter or paste the folder containing your spectrum CSV files.",
            font=("TkDefaultFont", 12),
        ).pack(pady=(0, 25))

        path_frame = ttk.Frame(frame)
        path_frame.pack(fill="x", padx=50)

        self.path_entry = ttk.Entry(path_frame)
        self.path_entry.pack(side="left", fill="x", expand=True, ipady=6)
        ttk.Button(path_frame, text="Browse...", command=self._browse_directory).pack(side="left", padx=(10, 0))

        ttk.Button(frame, text="Load CSV Files", command=self._load_directory, width=25).pack(pady=30)

    def _browse_directory(self) -> None:
        directory = filedialog.askdirectory(title="Select spectrum data directory")
        if directory:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, directory)

    def _load_directory(self) -> None:
        raw_path = self.path_entry.get().strip()
        if not raw_path:
            messagebox.showerror("Error", "Please enter or select a data directory.")
            return

        directory = Path(raw_path)
        if not directory.is_dir():
            messagebox.showerror("Error", "The selected directory does not exist.")
            return

        self.data_dir = directory
        self.root.config(cursor="watch")
        self.root.update()
        try:
            self.all_spectra, warnings = load_all_spectra(directory)
            self.column_names = get_unique_column_names(self.all_spectra)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return
        finally:
            self.root.config(cursor="")

        for warning in warnings:
            log.warning(warning)

        self.plot_groups = []
        self.show_column_selection()

    # -----------------------------------------------------------------
    # Screen 2: pick columns for a group
    # -----------------------------------------------------------------

    def show_column_selection(self) -> None:
        self.clear_window()
        self.selected_column_names = set()

        outer = ttk.Frame(self.root, padding=20)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer, text=f"Select observations for Plot {len(self.plot_groups) + 1}",
            font=("TkDefaultFont", 18, "bold"),
        ).pack(pady=(5, 5))
        ttk.Label(outer, text="Click observations to select them. Click again to deselect.").pack(pady=(0, 15))

        self.selection_count_label = ttk.Label(outer, text="Selected: 0")
        self.selection_count_label.pack(pady=(0, 10))

        content_frame = self._make_scrollable_frame(outer)
        self.column_buttons_by_name = {}
        for index, name in enumerate(self.column_names):
            button = tk.Button(
                content_frame, text=name, width=30, height=2, wraplength=220,
                relief="raised", bd=2, command=lambda n=name: self._toggle_column(n),
            )
            button.grid(row=index // BUTTON_COLUMNS, column=index % BUTTON_COLUMNS, padx=8, pady=6, sticky="ew")
            self.column_buttons_by_name[name] = button

        bottom_bar = ttk.Frame(outer)
        bottom_bar.pack(fill="x", pady=(15, 0))
        ttk.Button(bottom_bar, text="Back", command=self._go_back_from_selection).pack(side="left")
        ttk.Button(bottom_bar, text="CONFIRM SELECTION", command=self._confirm_selection).pack(side="right")

    def _toggle_column(self, name: str) -> None:
        button = self.column_buttons_by_name[name]
        if name in self.selected_column_names:
            self.selected_column_names.remove(name)
            button.config(relief="raised", bg="SystemButtonFace")
        else:
            self.selected_column_names.add(name)
            button.config(relief="sunken", bg="#b8d7e5")
        self.selection_count_label.config(text=f"Selected: {len(self.selected_column_names)}")

    def _confirm_selection(self) -> None:
        if not self.selected_column_names:
            messagebox.showwarning("No selection", "Please select at least one observation.")
            return
        selected_names = sorted(self.selected_column_names, key=str.lower)
        self.show_customization_screen(selected_names)

    def _go_back_from_selection(self) -> None:
        if self.plot_groups:
            self.plot_groups.pop()
            if self.plot_groups:
                self._ask_another_group()
                return
        self.show_directory_screen()

    # -----------------------------------------------------------------
    # Screen 3: name the group, then name & colour each observation in it
    # -----------------------------------------------------------------

    def show_customization_screen(self, selected_names: list[str]) -> None:
        self.clear_window()
        ttk.Label(self.root, text="Customize this plot group", font=("TkDefaultFont", 18, "bold")).pack(pady=(20, 5))
        ttk.Label(
            self.root,
            text="Give this plot its own title, then choose the name and colour for each observation.",
        ).pack(pady=(0, 10))
        title_frame = ttk.Frame(self.root, padding=(30, 0))
        title_frame.pack(fill="x", pady=(0, 15))
        ttk.Label(title_frame, text="Plot title:", font=("TkDefaultFont", 10, "bold")).pack(side="left")
        self.group_title_entry = ttk.Entry(title_frame, width=45)
        self.group_title_entry.pack(side="left", padx=10, ipady=3)

        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True, padx=30)
        content_frame = self._make_scrollable_frame(outer)

        self.customization_name_entries = []
        self.customization_color_entries = []

        for index, column_name in enumerate(selected_names):
            row = tk.Frame(content_frame, bd=1, relief="groove", padx=10, pady=10)
            row.pack(fill="x", pady=5)

            tk.Label(
                row, text=column_name, width=28, anchor="w", font=("TkDefaultFont", 10, "bold"),
            ).pack(side="left")

            display_name_entry = ttk.Entry(row, width=28)
            display_name_entry.insert(0, column_name)
            display_name_entry.pack(side="left", padx=10)

            default_color = COLOR_PALETTE[index % len(COLOR_PALETTE)]
            color_button = tk.Button(
                row, text="Choose colour", bg=default_color, width=15,
                command=lambda i=index: self._choose_color(i),
            )
            color_button.pack(side="left", padx=10)
            color_hex_label = tk.Label(row, text=default_color, width=10)
            color_hex_label.pack(side="left")

            self.customization_name_entries.append(display_name_entry)
            self.customization_color_entries.append({
                "column_name": column_name,
                "color": default_color,
                "button": color_button,
                "label": color_hex_label,
            })

        bottom_bar = ttk.Frame(self.root)
        bottom_bar.pack(pady=15)
        ttk.Button(bottom_bar, text="Back", command=self.show_column_selection).pack(side="left", padx=10)
        ttk.Button(bottom_bar, text="Confirm Names & Colours", command=self._confirm_customization).pack(side="left", padx=10)

    def _choose_color(self, index: int) -> None:
        current_color = self.customization_color_entries[index]["color"]
        chosen = colorchooser.askcolor(color=current_color, title="Choose spectrum colour")
        if chosen[1] is None:
            return
        new_color = chosen[1]
        entry = self.customization_color_entries[index]
        entry["color"] = new_color
        entry["button"].config(bg=new_color)
        entry["label"].config(text=new_color)

    def _confirm_customization(self) -> None:
        group_title = self.group_title_entry.get().strip()
        if not group_title:
            messagebox.showwarning("Missing title", "Please enter a title for this plot.")
            return

        group_items: list[GroupItem] = []
        for index, color_entry in enumerate(self.customization_color_entries):
            display_name = self.customization_name_entries[index].get().strip()
            if not display_name:
                messagebox.showwarning(
                    "Missing display name",
                    f"Please enter a display name for:\n\n{color_entry['column_name']}",
                )
                return
            group_items.append(
                GroupItem(color_entry["column_name"], display_name, color_entry["color"])
            )

        self.plot_groups.append(PlotGroup(title=group_title, items=group_items))
        self._ask_another_group()

    def _ask_another_group(self) -> None:
        if messagebox.askyesno("Another group?", "Do you want to create another group of observations?"):
            self.show_column_selection()
        else:
            self.show_wavelength_range_screen()

    # -----------------------------------------------------------------
    # Screen 4: wavelength range (shared by every group)
    # -----------------------------------------------------------------

    def show_wavelength_range_screen(self) -> None:
        self.clear_window()
        frame = ttk.Frame(self.root, padding=40)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Wavelength range", font=("TkDefaultFont", 18, "bold")).pack(pady=(20, 30))

        ttk.Label(frame, text="Lowest wavelength (nm):").pack()
        self.min_wavelength_entry = ttk.Entry(frame, width=25)
        self.min_wavelength_entry.pack(pady=(5, 25), ipady=5)

        ttk.Label(frame, text="Highest wavelength (nm):").pack()
        self.max_wavelength_entry = ttk.Entry(frame, width=25)
        self.max_wavelength_entry.pack(pady=(5, 30), ipady=5)

        button_frame = ttk.Frame(frame)
        button_frame.pack()
        ttk.Button(button_frame, text="Back", command=self.show_column_selection).pack(side="left", padx=10)
        ttk.Button(button_frame, text="Continue", command=self._validate_wavelength_range).pack(side="left", padx=10)

    def _validate_wavelength_range(self) -> None:
        try:
            wavelength_min_nm = float(self.min_wavelength_entry.get().strip().replace(",", "."))
            wavelength_max_nm = float(self.max_wavelength_entry.get().strip().replace(",", "."))
        except ValueError:
            messagebox.showerror("Invalid wavelength", "Please enter numeric wavelength values.")
            return

        if wavelength_min_nm >= wavelength_max_nm:
            messagebox.showerror("Invalid range", "The lowest wavelength must be smaller than the highest wavelength.")
            return

        self.wavelength_min_nm = wavelength_min_nm
        self.wavelength_max_nm = wavelength_max_nm
        self.show_normalization_screen()

    # -----------------------------------------------------------------
    # Screen 5: choose raw / AUC / both
    # -----------------------------------------------------------------

    def show_normalization_screen(self) -> None:
        self.clear_window()
        frame = ttk.Frame(self.root, padding=40)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Plot type", font=("TkDefaultFont", 18, "bold")).pack(pady=(20, 20))
        ttk.Label(frame, text="Which plots would you like to create?").pack(pady=(0, 20))

        self.plot_type = tk.StringVar(value="both")
        for label, value in (("Raw", "raw"), ("AUC normalized", "auc"), ("Both", "both")):
            ttk.Radiobutton(frame, text=label, variable=self.plot_type, value=value).pack(pady=5)

        group_titles = ", ".join(group.title for group in self.plot_groups)
        ttk.Label(
            frame,
            text=(
                f"\nGroups created: {len(self.plot_groups)} ({group_titles})\n"
                f"Wavelength range: {self.wavelength_min_nm:g} \u2013 {self.wavelength_max_nm:g} nm"
            ),
            justify="center",
        ).pack(pady=25)

        button_frame = ttk.Frame(frame)
        button_frame.pack()
        ttk.Button(button_frame, text="Back", command=self.show_wavelength_range_screen).pack(side="left", padx=10)
        ttk.Button(button_frame, text="GENERATE PLOTS", command=self._generate_plots).pack(side="left", padx=10)

    # -----------------------------------------------------------------
    # Plot generation
    # -----------------------------------------------------------------

    def _spectra_for_column(self, column_name: str) -> list[Spectrum]:
        return [spectrum for spectrum in self.all_spectra if spectrum.name == column_name]

    def _resolve_group_items(self, group: PlotGroup) -> list[tuple[Spectrum, GroupItem]]:
        """Expand a group's column selections into (spectrum, item) pairs,
        dropping any spectra that are exact data duplicates."""
        candidate_pairs = [
            (spectrum, item)
            for item in group.items
            for spectrum in self._spectra_for_column(item.column_name)
        ]

        seen_keys: set[tuple] = set()
        unique_pairs: list[tuple[Spectrum, GroupItem]] = []
        for spectrum, item in candidate_pairs:
            key = spectrum.duplicate_key()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_pairs.append((spectrum, item))
        return unique_pairs

    def _generate_plots(self) -> None:
        plot_type = self.plot_type.get()
        raw_output_dir = self.data_dir / RAW_FOLDER_NAME
        auc_output_dir = self.data_dir / AUC_FOLDER_NAME

        if plot_type in ("raw", "both"):
            raw_output_dir.mkdir(exist_ok=True)
        if plot_type in ("auc", "both"):
            auc_output_dir.mkdir(exist_ok=True)

        generated_group_count = 0
        self.root.config(cursor="watch")
        self.root.update()
        try:
            for group_number, group in enumerate(self.plot_groups, start=1):
                observation_pairs = self._resolve_group_items(group)
                if not observation_pairs:
                    log.warning(f"Plot group {group_number} ('{group.title}') contains no valid spectra.")
                    continue

                log.info(f"Generating Plot {group_number} ('{group.title}', {len(observation_pairs)} observations)")

                if plot_type in ("raw", "both"):
                    self._create_plot(observation_pairs, group, raw_output_dir, group_number, normalize=False)
                if plot_type in ("auc", "both"):
                    self._create_plot(observation_pairs, group, auc_output_dir, group_number, normalize=True)

                generated_group_count += 1
        finally:
            self.root.config(cursor="")

        self.show_finished_screen(generated_group_count, plot_type)

    def _create_plot(
        self,
        observation_pairs: list[tuple[Spectrum, GroupItem]],
        group: PlotGroup,
        output_dir: Path,
        group_number: int,
        *,
        normalize: bool,
    ) -> None:
        """Draw and save one figure, either raw radiance or AUC-normalized."""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

        for spectrum, item in observation_pairs:
            in_range_mask = (
                (spectrum.wavelength_nm >= self.wavelength_min_nm)
                & (spectrum.wavelength_nm <= self.wavelength_max_nm)
            )
            wavelength_window = spectrum.wavelength_nm[in_range_mask]
            radiance_window = spectrum.radiance[in_range_mask]
            if len(wavelength_window) < 2:
                continue

            if normalize:
                area_under_curve = np.trapezoid(radiance_window, wavelength_window)
                if area_under_curve == 0 or not np.isfinite(area_under_curve):
                    log.warning(f"Invalid AUC for {item.display_name} ({spectrum.source_file})")
                    continue
                radiance_window = radiance_window / area_under_curve

            ax.plot(
                wavelength_window, radiance_window,
                color=item.color, linewidth=1.6, alpha=0.9, label=item.display_name,
            )

        ax.set_xlabel("Wavelength (nm)", fontsize=12)
        ax.set_ylabel(
            "AUC-normalized radiance" if normalize else "Radiance [W/(sr\u00b7m\u00b2\u00b7nm)]",
            fontsize=12,
        )
        ax.set_title(group.title, fontsize=14, fontweight="bold")
        ax.set_xlim(self.wavelength_min_nm, self.wavelength_max_nm)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
        fig.tight_layout()

        title_slug = slugify_for_filename(group.title)
        filename = (
            f"Plot_{group_number}_{title_slug}_"
            f"{self.wavelength_min_nm:g}-{self.wavelength_max_nm:g}nm.png"
        )
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        kind_label = "AUC" if normalize else "RAW"
        log.info(f"Saved {kind_label}: {output_path}")

    # -----------------------------------------------------------------
    # Screen 6: done
    # -----------------------------------------------------------------

    def show_finished_screen(self, generated_group_count: int, plot_type: str) -> None:
        self.clear_window()
        frame = ttk.Frame(self.root, padding=40)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Plots successfully generated!", font=("TkDefaultFont", 18, "bold")).pack(pady=(40, 20))

        raw_status_text = "Created" if plot_type in ("raw", "both") else "Not created"
        auc_status_text = "Created" if plot_type in ("auc", "both") else "Not created"

        ttk.Label(
            frame,
            text=(
                f"Plot groups generated: {generated_group_count}\n\n"
                f"RAW plots: {raw_status_text}\n"
                f"AUC plots: {auc_status_text}"
            ),
            justify="center",
            font=("TkDefaultFont", 11),
        ).pack(pady=20)

        ttk.Button(frame, text="EXIT", command=self.root.destroy, width=20).pack(pady=30, ipady=5)


# =====================================================================
# Entry point
# =====================================================================

def main() -> None:
    root = tk.Tk()
    SpectrumPlotterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
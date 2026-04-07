#!/usr/bin/env python3
"""
Treatment Planner v2
====================
All features of treatment_planner.py, plus:

  7. CT Calibration  – estimate RescaleSlope / RescaleIntercept from air &
     water/tissue histogram peaks, preview the result, save a calibrated copy
     of the DICOM series to a new folder, and reload it ready for TOPAS.

Usage
-----
    python3 treatment_planner_v2.py

Coordinate system: CT volume [Z, Y, X].  Z index 0 = most-inferior slice.
"""

import os
import glob
import json
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, colorchooser, ttk
from datetime import datetime

import numpy as np
import pydicom
from pydicom.uid import generate_uid
from matplotlib.path import Path as MplPath

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# ═══════════════════════════════════════════════════════════════════════════
# Eclipse-style dark theme palette
# ═══════════════════════════════════════════════════════════════════════════

BG      = "#1e1e1e"
BG2     = "#252526"
BG3     = "#2d2d30"
FG      = "#d4d4d4"
FG_DIM  = "#858585"
ACCENT  = "#007acc"
ACCENT2 = "#4ec9b0"
BORDER  = "#3e3e42"
SEL     = "#094771"
XHAIR   = "#00e5ff"
AX_BG   = "#0d0d0d"


# ═══════════════════════════════════════════════════════════════════════════
# DICOM / file loaders
# ═══════════════════════════════════════════════════════════════════════════

def _collect_candidate_files(folder):
    """Return all files in folder (and one level of subdirs) that might be DICOM."""
    candidates = []
    for entry in sorted(os.scandir(folder), key=lambda e: e.name):
        if entry.is_file():
            candidates.append(entry.path)
        elif entry.is_dir():
            for sub in sorted(os.scandir(entry.path), key=lambda e: e.name):
                if sub.is_file():
                    candidates.append(sub.path)
    return candidates


def load_ct_series(folder, progress_cb=None):
    """Load CT DICOM series.

    Returns
    -------
    vol : ndarray [Z, Y, X] float32 HU
    meta : dict
    ordered_fpaths : list of absolute file paths in Z-ascending order
    """
    candidates = _collect_candidate_files(folder)
    if not candidates:
        raise FileNotFoundError(f"No files found in {folder}")

    n = len(candidates)
    if progress_cb:
        progress_cb(f"Scanning {n} files…")

    slice_info = []   # (z_mm, fpath, ds)
    for i, fpath in enumerate(candidates):
        if progress_cb and i % 10 == 0:
            progress_cb(f"Scanning {i + 1} / {n}…")
        try:
            ds = pydicom.dcmread(fpath, force=True)
            if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "PixelSpacing"):
                z = float(ds.ImagePositionPatient[2])
                slice_info.append((z, fpath, ds))
        except Exception:
            pass

    if not slice_info:
        raise FileNotFoundError(
            f"No valid CT DICOM slices found in:\n{folder}\n\n"
            "Make sure you select the folder that contains the .dcm files "
            "(or a parent folder with one level of subfolders).")

    slice_info.sort(key=lambda x: x[0])
    ordered_fpaths = [x[1] for x in slice_info]
    first_ds = slice_info[0][2]

    ps = [float(first_ds.PixelSpacing[0]), float(first_ds.PixelSpacing[1])]
    st = (abs(slice_info[1][0] - slice_info[0][0]) if len(slice_info) > 1
          else float(getattr(first_ds, "SliceThickness", 1.0)) or 1.0)
    try:
        rs, ri = float(first_ds.RescaleSlope), float(first_ds.RescaleIntercept)
    except AttributeError:
        rs, ri = 1.0, 0.0

    n_sl = len(slice_info)
    arrays = []
    for i, (_, _, ds) in enumerate(slice_info):
        if progress_cb:
            progress_cb(f"Stacking slice {i + 1} / {n_sl}…")
        arrays.append(ds.pixel_array)

    vol = np.stack(arrays).astype(np.float32) * rs + ri
    meta = dict(
        pixel_spacing=ps,
        slice_thickness=st,
        image_position=[float(v) for v in first_ds.ImagePositionPatient],
        slice_positions=[x[0] for x in slice_info],
    )
    return vol, meta, ordered_fpaths


# ═══════════════════════════════════════════════════════════════════════════
# CT Calibration helper
# ═══════════════════════════════════════════════════════════════════════════

def estimate_calibration_from_files(fpaths, sample_count=15, bins=300,
                                    air_hu=-1000.0, tissue_hu=0.0,
                                    progress_cb=None):
    """
    Sample raw stored pixel values from a subset of DICOM files and locate
    the air peak and the water/tissue peak by histogram binning.

    The two identified peaks are used as reference points for a linear
    rescale:  HU = slope * raw_pixel + intercept

    Returns
    -------
    slope, intercept : float
    air_raw, tiss_raw : float  (detected peak positions in raw pixel units)
    counts, centers   : ndarray  (histogram data for plotting)

    Raises
    ------
    ValueError if peaks cannot be located reliably.
    """
    step = max(1, len(fpaths) // sample_count)
    sample = fpaths[::step][:sample_count]

    pixels_list = []
    for i, fp in enumerate(sample):
        if progress_cb:
            progress_cb(f"Reading slice {i + 1} / {len(sample)} for calibration…")
        ds = pydicom.dcmread(fp, force=True)
        pixels_list.append(ds.pixel_array.flatten().astype(np.float32))

    pixels = np.concatenate(pixels_list)
    counts, edges = np.histogram(pixels, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    data_min = float(centers[0])
    data_max = float(centers[-1])

    # Auto-select thresholds based on whether data looks signed (HU-like) or unsigned
    if data_min >= -50:
        # Likely unsigned range (e.g. 0–4095 from small-animal scanners)
        span = data_max - data_min
        air_thresh = data_min + span * 0.25
        tiss_lo    = air_thresh
        tiss_hi    = data_min + span * 0.85
    else:
        # Signed, already HU-like stored values
        air_thresh = -500.0
        tiss_lo    = -500.0
        tiss_hi    =  800.0

    air_mask = centers < air_thresh
    if not np.any(air_mask):
        raise ValueError(
            f"No air peak found below threshold {air_thresh:.0f}.\n"
            "The field of view may not include air voxels, or the raw pixel\n"
            "range is unexpected. Try adjusting 'Air thresh' manually.")

    air_raw = float(centers[air_mask][np.argmax(counts[air_mask])])

    tiss_mask = (centers > tiss_lo) & (centers < tiss_hi)
    if not np.any(tiss_mask):
        raise ValueError(
            f"No tissue peak found between {tiss_lo:.0f} and {tiss_hi:.0f}.\n"
            "Check that soft tissue is present in the scan.")

    tiss_raw = float(centers[tiss_mask][np.argmax(counts[tiss_mask])])

    if abs(tiss_raw - air_raw) < 1.0:
        raise ValueError(
            f"Air ({air_raw:.1f}) and tissue ({tiss_raw:.1f}) peaks coincide — "
            "cannot compute a reliable calibration.")

    slope     = (tissue_hu - air_hu) / (tiss_raw - air_raw)
    intercept = tissue_hu - slope * tiss_raw

    return slope, intercept, air_raw, tiss_raw, counts, centers


# ═══════════════════════════════════════════════════════════════════════════
# Other loaders
# ═══════════════════════════════════════════════════════════════════════════

def load_rt_dose_dicom(path):
    """Load DICOM RT Dose → (volume [Z,Y,X] Gy, meta dict)."""
    ds = pydicom.dcmread(path, force=True)
    scaling = float(getattr(ds, "DoseGridScaling", 1.0))
    vol = ds.pixel_array.astype(np.float64) * scaling
    ps = [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])]
    if hasattr(ds, "GridFrameOffsetVector") and len(ds.GridFrameOffsetVector) > 1:
        offs = [float(v) for v in ds.GridFrameOffsetVector]
        st = abs(offs[1] - offs[0])
    else:
        st = float(getattr(ds, "SliceThickness", 2.0))
    ip = [float(v) for v in ds.ImagePositionPatient]
    return vol, dict(pixel_spacing=ps, slice_thickness=st, image_position=ip)


def load_topas_dose(header_path):
    """Load TOPAS binary dose output (.header + .bin).

    Returns (volume [Z,Y,X] Gy, meta dict) or raises.
    """
    params = {}
    with open(header_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if ":" in line:
                k, _, v = line.partition(":")
                params[k.strip()] = v.strip()

    nx = int(params.get("Number of voxels X", params.get("Nx", 1)))
    ny = int(params.get("Number of voxels Y", params.get("Ny", 1)))
    nz = int(params.get("Number of voxels Z", params.get("Nz", 1)))
    dx = float(params.get("Voxel size X (cm)", 0.1)) * 10.0
    dy = float(params.get("Voxel size Y (cm)", 0.1)) * 10.0
    dz = float(params.get("Voxel size Z (cm)", 0.1)) * 10.0

    stem = os.path.splitext(header_path)[0]
    bin_path = stem + ".bin"
    if not os.path.exists(bin_path):
        bin_path = stem + ".binheader"
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Cannot find binary file for {header_path}")

    raw = np.fromfile(bin_path, dtype=np.float32)
    vol = raw.reshape((nz, ny, nx))
    meta = dict(pixel_spacing=[dy, dx], slice_thickness=dz,
                image_position=[0.0, 0.0, 0.0])
    return vol, meta


def register_dose_to_ct(dose_vol, dose_meta, ct_meta, ct_shape):
    """Resample dose to CT grid via trilinear interpolation."""
    if dose_vol.shape == ct_shape:
        return dose_vol.astype(np.float32)
    try:
        from scipy.ndimage import zoom as ndimage_zoom
        nz_ct, ny_ct, nx_ct = ct_shape
        nz_d, ny_d, nx_d = dose_vol.shape
        zf = nz_ct / nz_d * (dose_meta["slice_thickness"] / ct_meta["slice_thickness"])
        yf = ny_ct / ny_d * (dose_meta["pixel_spacing"][0] / ct_meta["pixel_spacing"][0])
        xf = nx_ct / nx_d * (dose_meta["pixel_spacing"][1] / ct_meta["pixel_spacing"][1])
        r = ndimage_zoom(dose_vol, [zf, yf, xf], order=1, mode="constant", cval=0.0, prefilter=False)
        out = np.zeros(ct_shape, dtype=np.float32)
        sz = min(r.shape[0], ct_shape[0])
        sy = min(r.shape[1], ct_shape[1])
        sx = min(r.shape[2], ct_shape[2])
        out[:sz, :sy, :sx] = r[:sz, :sy, :sx]
        return out
    except Exception as exc:
        messagebox.showwarning("Dose registration",
                               f"Resampling failed: {exc}\nUsing dose as-is.")
        return dose_vol.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════

def polygon_x_intersections(pts, x_mm):
    result = []
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        if x1 == x2:
            continue
        if min(x1, x2) <= x_mm <= max(x1, x2):
            t = (x_mm - x1) / (x2 - x1)
            result.append(y1 + t * (y2 - y1))
    return sorted(result)


def polygon_y_intersections(pts, y_mm):
    result = []
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        if y1 == y2:
            continue
        if min(y1, y2) <= y_mm <= max(y1, y2):
            t = (y_mm - y1) / (y2 - y1)
            result.append(x1 + t * (x2 - x1))
    return sorted(result)


def fill_polygon_mask(pts_mm, ny, nx, row_mm, col_mm):
    if len(pts_mm) < 3:
        return np.zeros((ny, nx), dtype=bool)
    path = MplPath(pts_mm)
    ys = (np.arange(ny) + 0.5) * row_mm
    xs = (np.arange(nx) + 0.5) * col_mm
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    return path.contains_points(pts).reshape(ny, nx)


# ═══════════════════════════════════════════════════════════════════════════
# DVH
# ═══════════════════════════════════════════════════════════════════════════

def compute_dvh(dose_vol, contours, ct_meta):
    row_mm = ct_meta["pixel_spacing"][0]
    col_mm = ct_meta["pixel_spacing"][1]
    st_mm  = ct_meta["slice_thickness"]
    nz, ny, nx = dose_vol.shape
    voxel_vol_cc = row_mm * col_mm * st_mm / 1000.0

    all_d = []
    for z_idx, poly_list in contours.items():
        if z_idx < 0 or z_idx >= nz:
            continue
        mask = np.zeros((ny, nx), dtype=bool)
        for pts in poly_list:
            mask |= fill_polygon_mask(pts, ny, nx, row_mm, col_mm)
        all_d.extend(dose_vol[z_idx][mask].tolist())

    if not all_d:
        return np.array([0.0, 0.0]), np.array([100.0, 0.0]), {}

    d = np.sort(all_d)
    n = len(d)
    v = np.linspace(100.0, 0.0, n)
    doses = np.concatenate([[0.0], d])
    vols  = np.concatenate([[100.0], v])

    stats = {
        "Dmin":   float(np.min(d)),
        "Dmean":  float(np.mean(d)),
        "Dmax":   float(np.max(d)),
        "D95":    float(np.percentile(d, 5)),
        "D50":    float(np.percentile(d, 50)),
        "vol_cc": n * voxel_vol_cc,
    }
    return doses, vols, stats


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f1c40f",
    "#9b59b6", "#1abc9c", "#e67e22", "#e91e63",
    "#00bcd4", "#8bc34a",
]


# ═══════════════════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════════════════

class TreatmentPlanner:
    def __init__(self, master):
        self.master = master
        master.title("Treatment Planner v2")
        master.configure(bg=BG)
        master.bind("<Key>", self._on_key)
        self._apply_dark_theme()

        # ── CT ────────────────────────────────────────────────────────
        self.ct_volume = None
        self.ct_meta   = None
        self.ct_folder = None          # source folder (needed for calibration)
        self.ct_fpaths = []            # ordered file paths (Z-ascending)

        # ── CT Calibration ────────────────────────────────────────────
        self.calib_slope      = None
        self.calib_intercept  = None
        self.calib_air_raw    = None
        self.calib_tiss_raw   = None
        self.calib_hist_data  = None   # (counts, centers) for histogram popup
        self.calib_out_folder = None   # last saved calibrated folder

        # ── Dose ──────────────────────────────────────────────────────
        self.dose_volume   = None
        self.dose_visible  = True
        self.dose_opacity  = 0.5
        self.dose_thr_frac = 0.05

        # ── Navigation ────────────────────────────────────────────────
        self.cur_x = 0
        self.cur_y = 0
        self.cur_z = 0

        # ── W/L ───────────────────────────────────────────────────────
        self.win_width = 2000.0
        self.win_level = 200.0

        # ── Structures ────────────────────────────────────────────────
        self.structures   = {}
        self.struct_order = []
        self.active_struct = None

        # ── Drawing ───────────────────────────────────────────────────
        self.draw_mode      = False
        self._draw_pts      = []
        self._draw_mouse_mm = None

        # ── TOPAS ─────────────────────────────────────────────────────
        self._topas_macro = None

        self._cbar         = None
        self._motion_after = None
        self._dvh_cache    = {}
        self._build_gui()

    # ══════════════════════════════════════════════════════════════════
    # Dark theme
    # ══════════════════════════════════════════════════════════════════

    def _apply_dark_theme(self):
        style = ttk.Style(self.master)
        style.theme_use("clam")
        style.configure(".",
                        background=BG, foreground=FG,
                        font=("Helvetica", 10), relief="flat")
        style.configure("TFrame",      background=BG)
        style.configure("TLabel",      background=BG, foreground=FG)
        style.configure("Dim.TLabel",  background=BG, foreground=FG_DIM,
                        font=("Helvetica", 9))
        style.configure("Head.TLabel", background=BG, foreground=ACCENT,
                        font=("Helvetica", 10, "bold"))
        style.configure("Pos.TLabel",  background=BG, foreground=ACCENT2,
                        font=("Helvetica", 9))
        style.configure("OK.TLabel",   background=BG, foreground="#2ecc71",
                        font=("Helvetica", 9))
        style.configure("TButton",
                        background=BG3, foreground=FG,
                        borderwidth=1, relief="flat", padding=(6, 3))
        style.map("TButton",
                  background=[("active", ACCENT), ("pressed", "#005999")],
                  foreground=[("active", "white"), ("pressed", "white")])
        style.configure("TEntry",
                        fieldbackground=BG3, foreground=FG,
                        insertcolor=FG, borderwidth=1, relief="flat",
                        selectbackground=SEL, selectforeground=FG)
        style.configure("TCheckbutton", background=BG, foreground=FG)
        style.map("TCheckbutton",
                  background=[("active", BG)],
                  foreground=[("active", FG)])
        style.configure("TSeparator",  background=BORDER)
        style.configure("TScrollbar",
                        background=BG2, troughcolor=BG3,
                        bordercolor=BORDER, arrowcolor=FG_DIM,
                        relief="flat", gripcount=0)
        style.map("TScrollbar", background=[("active", BG3)])
        style.configure("TPanedwindow", background=BORDER)

    # ══════════════════════════════════════════════════════════════════
    # GUI
    # ══════════════════════════════════════════════════════════════════

    def _build_gui(self):
        paned = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # ── left scrollable controls ────────────────────────────────
        lo = ttk.Frame(paned)
        paned.add(lo, weight=0)
        cc = tk.Canvas(lo, highlightthickness=0, width=310, bg=BG2)
        sb = ttk.Scrollbar(lo, orient=tk.VERTICAL, command=cc.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        cc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cc.configure(yscrollcommand=sb.set)
        cf = ttk.Frame(cc, padding=(8, 4))
        wid = cc.create_window((0, 0), window=cf, anchor="nw")
        cc.bind("<Configure>", lambda e: cc.itemconfig(wid, width=e.width))
        cf.bind("<Configure>", lambda e: cc.configure(scrollregion=cc.bbox("all")))
        def _mw(e): cc.yview_scroll(int(-1 * (e.delta / 120)), "units")
        cc.bind("<Enter>", lambda e: cc.bind_all("<MouseWheel>", _mw))
        cc.bind("<Leave>", lambda e: cc.unbind_all("<MouseWheel>"))

        # ── CT section ──────────────────────────────────────────────
        ttk.Label(cf, text="CT DATA", style="Head.TLabel").pack(anchor="w", pady=(8, 2))
        ttk.Button(cf, text="Load CT DICOM folder…",
                   command=self._load_ct).pack(fill=tk.X, pady=2)
        self.ct_info_var = tk.StringVar(value="No CT loaded")
        ttk.Label(cf, textvariable=self.ct_info_var, wraplength=285,
                  style="Dim.TLabel").pack(fill=tk.X)

        ttk.Separator(cf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # ── CT Calibration section ───────────────────────────────────
        ttk.Label(cf, text="CT CALIBRATION", style="Head.TLabel").pack(anchor="w", pady=(8, 2))
        ttk.Label(cf, text="Estimate RescaleSlope/Intercept from air & water\n"
                           "peaks in the raw pixel histogram.",
                  style="Dim.TLabel", wraplength=285).pack(fill=tk.X)

        # Config row 1: sample slices + histogram bins
        cr1 = ttk.Frame(cf); cr1.pack(fill=tk.X, pady=2)
        ttk.Label(cr1, text="Sample slices:", width=14).pack(side=tk.LEFT)
        self.calib_sample_var = tk.StringVar(value="15")
        ttk.Entry(cr1, textvariable=self.calib_sample_var, width=5).pack(side=tk.LEFT)
        ttk.Label(cr1, text="  Bins:", width=6).pack(side=tk.LEFT)
        self.calib_bins_var = tk.StringVar(value="300")
        ttk.Entry(cr1, textvariable=self.calib_bins_var, width=5).pack(side=tk.LEFT)

        # Config row 2: reference HU values
        cr2 = ttk.Frame(cf); cr2.pack(fill=tk.X, pady=2)
        ttk.Label(cr2, text="Air HU ref:", width=10).pack(side=tk.LEFT)
        self.calib_air_hu_var = tk.StringVar(value="-1000")
        ttk.Entry(cr2, textvariable=self.calib_air_hu_var, width=7).pack(side=tk.LEFT)
        ttk.Label(cr2, text="  Tissue HU:", width=10).pack(side=tk.LEFT)
        self.calib_tiss_hu_var = tk.StringVar(value="0")
        ttk.Entry(cr2, textvariable=self.calib_tiss_hu_var, width=7).pack(side=tk.LEFT)

        ttk.Button(cf, text="Estimate calibration from CT…",
                   command=self._estimate_calibration).pack(fill=tk.X, pady=(4, 2))

        # Calibration result readout
        self.calib_result_var = tk.StringVar(value="No calibration estimated yet.")
        ttk.Label(cf, textvariable=self.calib_result_var, wraplength=285,
                  style="Pos.TLabel").pack(fill=tk.X, pady=1)

        cr3 = ttk.Frame(cf); cr3.pack(fill=tk.X, pady=2)
        ttk.Button(cr3, text="Show histogram",
                   command=self._show_calib_histogram).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(cr3, text="Save calibrated DICOM…",
                   command=self._save_calibrated_dicom).pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.calib_save_var = tk.StringVar(value="")
        ttk.Label(cf, textvariable=self.calib_save_var, wraplength=285,
                  style="Dim.TLabel").pack(fill=tk.X, pady=1)

        ttk.Button(cf, text="Reload calibrated CT into viewer",
                   command=self._reload_calibrated_ct).pack(fill=tk.X, pady=2)

        ttk.Separator(cf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # ── Dose section ────────────────────────────────────────────
        ttk.Label(cf, text="DOSE DISTRIBUTION", style="Head.TLabel").pack(anchor="w", pady=(8, 2))
        ttk.Button(cf, text="Load DICOM RT Dose (.dcm)…",
                   command=self._load_dose_dicom).pack(fill=tk.X, pady=2)
        ttk.Button(cf, text="Load dose array (.npy)…",
                   command=self._load_dose_npy).pack(fill=tk.X, pady=2)
        self.dose_info_var = tk.StringVar(value="No dose loaded")
        ttk.Label(cf, textvariable=self.dose_info_var, wraplength=285,
                  style="Dim.TLabel").pack(fill=tk.X)

        opa_f = ttk.Frame(cf); opa_f.pack(fill=tk.X, pady=2)
        ttk.Label(opa_f, text="Opacity:", width=9).pack(side=tk.LEFT)
        self.opacity_var = tk.StringVar(value="0.50")
        ttk.Entry(opa_f, textvariable=self.opacity_var, width=6).pack(side=tk.LEFT)
        ttk.Button(opa_f, text="Set",
                   command=self._set_dose_opacity).pack(side=tk.LEFT, padx=4)
        self.dose_vis_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cf, text="Show dose overlay",
                        variable=self.dose_vis_var,
                        command=self._toggle_dose_vis).pack(anchor="w", pady=2)

        ttk.Separator(cf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # ── Window / Level ───────────────────────────────────────────
        ttk.Label(cf, text="WINDOW / LEVEL", style="Head.TLabel").pack(anchor="w", pady=(8, 2))
        wf = ttk.Frame(cf); wf.pack(fill=tk.X, pady=1)
        ttk.Label(wf, text="Width:", width=8).pack(side=tk.LEFT)
        self.ww_var = tk.StringVar(value="2000")
        ttk.Entry(wf, textvariable=self.ww_var, width=8).pack(side=tk.LEFT)
        lf = ttk.Frame(cf); lf.pack(fill=tk.X, pady=1)
        ttk.Label(lf, text="Level:", width=8).pack(side=tk.LEFT)
        self.wl_var = tk.StringVar(value="200")
        ttk.Entry(lf, textvariable=self.wl_var, width=8).pack(side=tk.LEFT)
        ttk.Button(cf, text="Apply W/L",
                   command=self._apply_wl).pack(fill=tk.X, pady=2)
        pf = ttk.Frame(cf); pf.pack(fill=tk.X, pady=2)
        for label, (w, l) in [("Soft tissue", (400, 40)),
                               ("Lung", (1500, -600)), ("Bone", (2000, 500))]:
            ttk.Button(pf, text=label,
                       command=lambda w=w, l=l: self._preset(w, l)
                       ).pack(side=tk.LEFT, expand=True, fill=tk.X)

        ttk.Separator(cf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # ── Structures ───────────────────────────────────────────────
        ttk.Label(cf, text="STRUCTURES (ROIs)", style="Head.TLabel").pack(anchor="w", pady=(8, 2))
        sf = ttk.Frame(cf); sf.pack(fill=tk.X)
        self.struct_lb = tk.Listbox(sf, height=7, selectmode=tk.SINGLE,
                                    exportselection=False, font=("Courier", 10),
                                    bg=BG3, fg=FG, selectbackground=SEL,
                                    selectforeground=FG, relief="flat",
                                    highlightthickness=1, highlightbackground=BORDER,
                                    highlightcolor=ACCENT, activestyle="none",
                                    borderwidth=0)
        lbsb = ttk.Scrollbar(sf, orient=tk.VERTICAL, command=self.struct_lb.yview)
        self.struct_lb.configure(yscrollcommand=lbsb.set)
        self.struct_lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lbsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.struct_lb.bind("<<ListboxSelect>>", self._on_struct_select)
        self.struct_lb.bind("<Double-1>", lambda e: self._rename_structure())

        self.color_canvas = tk.Canvas(cf, height=22, highlightthickness=1,
                                      highlightbackground=BORDER, bg=BG3,
                                      cursor="hand2")
        self.color_canvas.pack(fill=tk.X, pady=2)
        self.color_canvas.bind("<Button-1>", lambda e: self._change_struct_color())

        sbr = ttk.Frame(cf); sbr.pack(fill=tk.X, pady=2)
        ttk.Button(sbr, text="+ Add",   command=self._add_structure
                   ).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(sbr, text="Rename",  command=self._rename_structure
                   ).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(sbr, text="Delete",  command=self._delete_structure
                   ).pack(side=tk.LEFT, expand=True, fill=tk.X)

        ttk.Separator(cf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # ── Contouring ───────────────────────────────────────────────
        ttk.Label(cf, text="CONTOURING", style="Head.TLabel").pack(anchor="w", pady=(8, 2))
        ttk.Label(cf, text="Select a structure. Toggle Draw, click axial\n"
                           "view to add points. Close to finish.",
                  style="Dim.TLabel", wraplength=285).pack(fill=tk.X)

        self.draw_mode_var = tk.BooleanVar(value=False)
        self.draw_chk = ttk.Checkbutton(cf, text="Draw mode (OFF)",
                                         variable=self.draw_mode_var,
                                         command=self._toggle_draw_mode)
        self.draw_chk.pack(anchor="w", pady=3)

        cbf = ttk.Frame(cf); cbf.pack(fill=tk.X, pady=2)
        ttk.Button(cbf, text="Close polygon",
                   command=self._close_contour).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(cbf, text="Cancel",
                   command=self._cancel_draw).pack(side=tk.LEFT, expand=True, fill=tk.X)

        c2 = ttk.Frame(cf); c2.pack(fill=tk.X, pady=2)
        ttk.Button(c2, text="Clear this slice",
                   command=self._clear_slice).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(c2, text="Clear all contours",
                   command=self._clear_all).pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.contour_info_var = tk.StringVar(value="")
        ttk.Label(cf, textvariable=self.contour_info_var, wraplength=285,
                  style="Pos.TLabel").pack(fill=tk.X, pady=2)

        ttk.Separator(cf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # ── Navigation ───────────────────────────────────────────────
        ttk.Label(cf, text="NAVIGATION", style="Head.TLabel").pack(anchor="w", pady=(8, 2))

        ax_nav = ttk.Frame(cf); ax_nav.pack(fill=tk.X, pady=1)
        ttk.Label(ax_nav, text="Axial Z:", width=10).pack(side=tk.LEFT)
        ttk.Button(ax_nav, text="◀", width=3,
                   command=lambda: self._step_z(-1)).pack(side=tk.LEFT)
        self.z_var = tk.StringVar(value="0")
        ttk.Entry(ax_nav, textvariable=self.z_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(ax_nav, text="▶", width=3,
                   command=lambda: self._step_z(1)).pack(side=tk.LEFT)
        ttk.Button(ax_nav, text="Go",
                   command=self._go_z).pack(side=tk.LEFT, padx=4)

        sg_nav = ttk.Frame(cf); sg_nav.pack(fill=tk.X, pady=1)
        ttk.Label(sg_nav, text="Sagittal X:", width=10).pack(side=tk.LEFT)
        ttk.Button(sg_nav, text="◀", width=3,
                   command=lambda: self._step_x(-1)).pack(side=tk.LEFT)
        self.x_var = tk.StringVar(value="0")
        ttk.Entry(sg_nav, textvariable=self.x_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(sg_nav, text="▶", width=3,
                   command=lambda: self._step_x(1)).pack(side=tk.LEFT)
        ttk.Button(sg_nav, text="Go",
                   command=self._go_x).pack(side=tk.LEFT, padx=4)

        co_nav = ttk.Frame(cf); co_nav.pack(fill=tk.X, pady=1)
        ttk.Label(co_nav, text="Coronal Y:", width=10).pack(side=tk.LEFT)
        ttk.Button(co_nav, text="◀", width=3,
                   command=lambda: self._step_y(-1)).pack(side=tk.LEFT)
        self.y_var = tk.StringVar(value="0")
        ttk.Entry(co_nav, textvariable=self.y_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(co_nav, text="▶", width=3,
                   command=lambda: self._step_y(1)).pack(side=tk.LEFT)
        ttk.Button(co_nav, text="Go",
                   command=self._go_y).pack(side=tk.LEFT, padx=4)

        self.pos_var = tk.StringVar(value="")
        ttk.Label(cf, textvariable=self.pos_var, wraplength=285,
                  style="Pos.TLabel").pack(fill=tk.X, pady=2)

        ttk.Separator(cf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # ── Save / Load ──────────────────────────────────────────────
        ttk.Label(cf, text="SAVE / LOAD", style="Head.TLabel").pack(anchor="w", pady=(8, 2))
        sc = ttk.Frame(cf); sc.pack(fill=tk.X, pady=2)
        ttk.Button(sc, text="Save contours (JSON)…",
                   command=self._save_contours).pack(fill=tk.X, pady=1)
        ttk.Button(sc, text="Load contours (JSON)…",
                   command=self._load_contours).pack(fill=tk.X, pady=1)
        ttk.Button(sc, text="Export view image…",
                   command=self._export_view).pack(fill=tk.X, pady=1)

        ttk.Separator(cf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # ── TOPAS ────────────────────────────────────────────────────
        ttk.Label(cf, text="TOPAS INTEGRATION", style="Head.TLabel").pack(anchor="w", pady=(8, 2))

        ttk.Label(cf, text="Active CT folder:", style="Dim.TLabel").pack(anchor="w")
        self.topas_ct_var = tk.StringVar(value="(no CT loaded)")
        ttk.Label(cf, textvariable=self.topas_ct_var, wraplength=285,
                  style="Pos.TLabel").pack(fill=tk.X)
        ttk.Button(cf, text="Copy CT folder path to clipboard",
                   command=self._copy_ct_path).pack(fill=tk.X, pady=(2, 4))

        ttk.Button(cf, text="Load TOPAS dose (.header)…",
                   command=self._load_topas_dose).pack(fill=tk.X, pady=1)
        ttk.Separator(cf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        ttk.Label(cf, text="Run TOPAS macro", style="Dim.TLabel").pack(anchor="w")

        tp_f = ttk.Frame(cf); tp_f.pack(fill=tk.X, pady=1)
        self.topas_macro_var = tk.StringVar(value="No macro selected")
        ttk.Label(tp_f, textvariable=self.topas_macro_var, wraplength=200,
                  style="Dim.TLabel").pack(side=tk.LEFT, fill=tk.X, expand=True)

        tp2 = ttk.Frame(cf); tp2.pack(fill=tk.X, pady=2)
        ttk.Button(tp2, text="Select macro…",
                   command=self._pick_topas_macro).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(tp2, text="Run TOPAS",
                   command=self._run_topas).pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.topas_status_var = tk.StringVar(value="")
        ttk.Label(cf, textvariable=self.topas_status_var, wraplength=285,
                  style="Dim.TLabel").pack(fill=tk.X)

        # ── right: 2×2 figure ────────────────────────────────────────
        right = tk.Frame(paned, bg=BG)
        paned.add(right, weight=1)

        self.fig = Figure(figsize=(14, 9), constrained_layout=True)
        self.fig.patch.set_facecolor(BG)
        gs = self.fig.add_gridspec(2, 2, hspace=0.06, wspace=0.06)
        self.ax_ax  = self.fig.add_subplot(gs[0, 0])
        self.ax_sag = self.fig.add_subplot(gs[0, 1])
        self.ax_cor = self.fig.add_subplot(gs[1, 0])
        self.ax_dvh = self.fig.add_subplot(gs[1, 1])

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        toolbar = NavigationToolbar2Tk(self.canvas, right)
        toolbar.config(bg=BG2)
        for w in toolbar.winfo_children():
            try: w.config(bg=BG2, fg=FG, activebackground=ACCENT, activeforeground="white")
            except Exception: pass
        toolbar.update()
        self.canvas.get_tk_widget().configure(bg=BG, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect("scroll_event",        self._on_scroll)
        self.canvas.mpl_connect("button_press_event",  self._on_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

        self._update_display()

    # ══════════════════════════════════════════════════════════════════
    # CT loading
    # ══════════════════════════════════════════════════════════════════

    def _load_ct(self):
        folder = filedialog.askdirectory(title="Select CT DICOM folder")
        if not folder:
            return
        self.ct_info_var.set("Loading CT…")

        def _progress(msg):
            self.master.after(0, lambda m=msg: self.ct_info_var.set(m))

        def _worker():
            try:
                vol, meta, fpaths = load_ct_series(folder, progress_cb=_progress)
                self.master.after(0, lambda: self._on_ct_loaded(vol, meta, folder, fpaths))
            except Exception as e:
                err = str(e)
                self.master.after(0, lambda: (
                    self.ct_info_var.set("Load failed."),
                    messagebox.showerror("CT Load Error", err)
                ))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_ct_loaded(self, vol, meta, folder, fpaths):
        self.ct_volume = vol
        self.ct_meta   = meta
        self.ct_folder = folder
        self.ct_fpaths = fpaths

        # Reset calibration state whenever a new CT is loaded
        self.calib_slope     = None
        self.calib_intercept = None
        self.calib_air_raw   = None
        self.calib_tiss_raw  = None
        self.calib_hist_data = None
        self.calib_result_var.set("No calibration estimated yet.")
        self.calib_save_var.set("")

        nz, ny, nx = vol.shape
        ps = meta["pixel_spacing"]
        st = meta["slice_thickness"]
        self.cur_z = nz // 2; self.cur_y = ny // 2; self.cur_x = nx // 2
        self.z_var.set(str(self.cur_z))
        self.y_var.set(str(self.cur_y))
        self.x_var.set(str(self.cur_x))
        self.ct_info_var.set(
            f"{os.path.basename(folder)}\n"
            f"{nz}×{ny}×{nx}  px={ps[0]:.3f}×{ps[1]:.3f} mm  sl={st:.3f} mm")
        self.topas_ct_var.set(folder)
        self._update_display()

    # ══════════════════════════════════════════════════════════════════
    # CT Calibration
    # ══════════════════════════════════════════════════════════════════

    def _estimate_calibration(self):
        if not self.ct_fpaths:
            messagebox.showwarning("Calibration", "Load a CT DICOM folder first.")
            return

        try:
            sample_count = int(self.calib_sample_var.get())
            bins         = int(self.calib_bins_var.get())
            air_hu       = float(self.calib_air_hu_var.get())
            tissue_hu    = float(self.calib_tiss_hu_var.get())
        except ValueError:
            messagebox.showerror("Calibration", "Invalid parameter values.")
            return

        self.calib_result_var.set("Estimating — please wait…")
        self.master.update_idletasks()

        def _progress(msg):
            self.master.after(0, lambda m=msg: self.calib_result_var.set(m))

        def _worker():
            try:
                slope, intercept, air_raw, tiss_raw, counts, centers = \
                    estimate_calibration_from_files(
                        self.ct_fpaths,
                        sample_count=sample_count,
                        bins=bins,
                        air_hu=air_hu,
                        tissue_hu=tissue_hu,
                        progress_cb=_progress,
                    )
                self.master.after(0, lambda: self._on_calib_done(
                    slope, intercept, air_raw, tiss_raw, counts, centers))
            except Exception as e:
                err = str(e)
                self.master.after(0, lambda: (
                    self.calib_result_var.set("Estimation failed."),
                    messagebox.showerror("Calibration Error", err)
                ))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_calib_done(self, slope, intercept, air_raw, tiss_raw, counts, centers):
        self.calib_slope     = slope
        self.calib_intercept = intercept
        self.calib_air_raw   = air_raw
        self.calib_tiss_raw  = tiss_raw
        self.calib_hist_data = (counts, centers)
        self.calib_result_var.set(
            f"Air raw={air_raw:.1f}  →  {self.calib_air_hu_var.get()} HU\n"
            f"Tissue raw={tiss_raw:.1f}  →  {self.calib_tiss_hu_var.get()} HU\n"
            f"Slope={slope:.5f}   Intercept={intercept:.2f}"
        )

    def _show_calib_histogram(self):
        if self.calib_hist_data is None:
            messagebox.showinfo("Histogram", "Run 'Estimate calibration' first.")
            return

        counts, centers = self.calib_hist_data
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(AX_BG)
        ax.plot(centers, counts, color=ACCENT2, lw=1.2)
        ax.set_xlabel("Raw stored pixel value", color=FG_DIM)
        ax.set_ylabel("Count", color=FG_DIM)
        ax.set_title("Calibration histogram — detected peaks", color=FG)
        ax.tick_params(colors=FG_DIM)
        for sp in ax.spines.values():
            sp.set_color(BORDER)

        if self.calib_air_raw is not None:
            ax.axvline(self.calib_air_raw,  color="#e74c3c", lw=1.5, linestyle="--",
                       label=f"Air peak  {self.calib_air_raw:.1f}")
        if self.calib_tiss_raw is not None:
            ax.axvline(self.calib_tiss_raw, color="#2ecc71", lw=1.5, linestyle="--",
                       label=f"Tissue peak  {self.calib_tiss_raw:.1f}")

        leg = ax.legend(facecolor=BG2, edgecolor=BORDER, labelcolor=FG, fontsize=9)
        fig.tight_layout()
        plt.show()

    def _save_calibrated_dicom(self):
        if self.calib_slope is None:
            messagebox.showwarning("Calibration", "Estimate calibration first.")
            return
        if not self.ct_fpaths or self.ct_folder is None:
            messagebox.showwarning("Calibration", "No CT data loaded.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_out = self.ct_folder.rstrip("/\\") + f"_calibrated_{timestamp}"

        out_folder = filedialog.askdirectory(
            title="Choose output folder for calibrated DICOM "
                  f"(default suggestion: …_calibrated_{timestamp})",
        )
        if not out_folder:
            out_folder = default_out

        os.makedirs(out_folder, exist_ok=True)

        self.calib_save_var.set("Saving calibrated DICOM…")
        self.master.update_idletasks()

        slope     = self.calib_slope
        intercept = self.calib_intercept
        air_hu    = float(self.calib_air_hu_var.get())
        tiss_hu   = float(self.calib_tiss_hu_var.get())
        new_series_uid = generate_uid()

        def _worker():
            n = len(self.ct_fpaths)
            log_lines = [
                "Calibration log",
                f"Timestamp   : {timestamp}",
                f"Source      : {self.ct_folder}",
                f"Output      : {out_folder}",
                f"Ref points  : air={air_hu:.0f} HU, tissue/water={tiss_hu:.0f} HU",
                f"Air peak    : {self.calib_air_raw:.2f} raw → {air_hu:.0f} HU",
                f"Tissue peak : {self.calib_tiss_raw:.2f} raw → {tiss_hu:.0f} HU",
                f"Slope       : {slope:.6f}",
                f"Intercept   : {intercept:.4f}",
                "-" * 60,
            ]

            try:
                for i, fpath in enumerate(self.ct_fpaths):
                    self.master.after(
                        0,
                        lambda i=i: self.calib_save_var.set(
                            f"Saving slice {i + 1} / {n}…")
                    )
                    ds = pydicom.dcmread(fpath, force=True)
                    ds.RescaleSlope     = slope
                    ds.RescaleIntercept = intercept
                    ds.RescaleType      = "HU"
                    ds.SeriesInstanceUID = new_series_uid

                    # Give each slice a unique SOP UID
                    ds.SOPInstanceUID = generate_uid()

                    # Sensible window defaults
                    ds.WindowCenter = int(tiss_hu)
                    ds.WindowWidth  = 400

                    out_path = os.path.join(out_folder, os.path.basename(fpath))
                    ds.save_as(out_path, write_like_original=False)

                log_path = os.path.join(out_folder, "calibration_log.txt")
                with open(log_path, "w") as fh:
                    fh.write("\n".join(log_lines) + "\n")

                self.master.after(0, lambda: self._on_save_done(out_folder))

            except Exception as e:
                err = str(e)
                self.master.after(0, lambda: (
                    self.calib_save_var.set("Save failed."),
                    messagebox.showerror("Save Error", err)
                ))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_save_done(self, out_folder):
        self.calib_out_folder = out_folder
        self.calib_save_var.set(f"Saved to:\n{out_folder}")
        if messagebox.askyesno(
                "Calibrated DICOM saved",
                f"Calibrated series saved to:\n{out_folder}\n\n"
                "Reload it into the viewer now?"):
            self._reload_calibrated_ct()

    def _reload_calibrated_ct(self):
        folder = self.calib_out_folder
        if not folder:
            folder = filedialog.askdirectory(
                title="Select calibrated CT DICOM folder to reload")
        if not folder:
            return

        self.ct_info_var.set("Loading calibrated CT…")
        self.master.update_idletasks()

        def _progress(msg):
            self.master.after(0, lambda m=msg: self.ct_info_var.set(m))

        def _worker():
            try:
                vol, meta, fpaths = load_ct_series(folder, progress_cb=_progress)
                self.master.after(0, lambda: self._on_ct_loaded(vol, meta, folder, fpaths))
            except Exception as e:
                err = str(e)
                self.master.after(0, lambda: (
                    self.ct_info_var.set("Reload failed."),
                    messagebox.showerror("CT Load Error", err)
                ))

        threading.Thread(target=_worker, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════
    # Dose loading
    # ══════════════════════════════════════════════════════════════════

    def _load_dose_dicom(self):
        path = filedialog.askopenfilename(
            title="Select DICOM RT Dose file",
            filetypes=[("DICOM", "*.dcm"), ("All", "*.*")])
        if not path:
            return
        try:
            vol, dmeta = load_rt_dose_dicom(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load RT dose:\n{e}")
            return
        self._set_dose(vol, dmeta, os.path.basename(path))

    def _load_dose_npy(self):
        path = filedialog.askopenfilename(
            title="Select dose numpy array",
            filetypes=[("NumPy", "*.npy"), ("All", "*.*")])
        if not path:
            return
        try:
            vol = np.load(path).astype(np.float32)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load .npy:\n{e}")
            return
        if vol.ndim != 3:
            messagebox.showerror("Error", f"Expected 3-D array, got shape {vol.shape}")
            return
        dmeta = self.ct_meta.copy() if self.ct_meta is not None else \
            dict(pixel_spacing=[1.0, 1.0], slice_thickness=1.0, image_position=[0, 0, 0])
        self._set_dose(vol, dmeta, os.path.basename(path))

    def _load_topas_dose(self):
        path = filedialog.askopenfilename(
            title="Select TOPAS dose header",
            filetypes=[("Header", "*.header"), ("All", "*.*")])
        if not path:
            return
        try:
            vol, dmeta = load_topas_dose(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load TOPAS dose:\n{e}")
            return
        self._set_dose(vol, dmeta, os.path.basename(path))

    def _set_dose(self, vol, dmeta, label):
        if self.ct_volume is not None:
            self.dose_volume = register_dose_to_ct(vol, dmeta, self.ct_meta, self.ct_volume.shape)
        else:
            self.dose_volume = vol.astype(np.float32)
        dmax = float(np.max(self.dose_volume))
        self.dose_info_var.set(
            f"{label}\n{self.dose_volume.shape}  max={dmax:.3f} Gy")
        if dmax > 0:
            peak = np.unravel_index(np.argmax(self.dose_volume), self.dose_volume.shape)
            self.cur_z = int(peak[0])
            self.cur_y = int(peak[1])
            self.cur_x = int(peak[2])
            self.z_var.set(str(self.cur_z))
            self.y_var.set(str(self.cur_y))
            self.x_var.set(str(self.cur_x))
        self._invalidate_dvh_cache()
        self._update_display()

    def _set_dose_opacity(self):
        try:
            self.dose_opacity = float(self.opacity_var.get())
        except ValueError:
            pass
        self._update_display()

    def _toggle_dose_vis(self):
        self.dose_visible = self.dose_vis_var.get()
        self._update_display()

    def _dose_rgba(self, dose_2d):
        dmax = float(np.max(self.dose_volume)) if self.dose_volume is not None else 1.0
        if dmax == 0:
            dmax = 1.0
        norm = np.clip(np.nan_to_num(dose_2d, nan=0.0) / dmax, 0.0, 1.0)
        rgba = np.array(plt.cm.jet(norm), dtype=np.float32)
        ramp = np.where(norm < self.dose_thr_frac, 0.0,
                        (norm - self.dose_thr_frac) / max(1.0 - self.dose_thr_frac, 1e-6))
        rgba[..., 3] = (ramp * self.dose_opacity).astype(np.float32)
        return rgba

    # ══════════════════════════════════════════════════════════════════
    # Window / Level
    # ══════════════════════════════════════════════════════════════════

    def _apply_wl(self):
        try:
            self.win_width = float(self.ww_var.get())
            self.win_level = float(self.wl_var.get())
        except ValueError:
            return
        self._update_display()

    def _preset(self, w, l):
        self.win_width = float(w); self.win_level = float(l)
        self.ww_var.set(str(w)); self.wl_var.set(str(l))
        self._update_display()

    # ══════════════════════════════════════════════════════════════════
    # Structures
    # ══════════════════════════════════════════════════════════════════

    def _add_structure(self):
        name = simpledialog.askstring("New structure", "Structure name:",
                                      parent=self.master)
        if not name or name.strip() == "":
            return
        name = name.strip()
        if name in self.structures:
            messagebox.showwarning("Exists", f"Structure '{name}' already exists.")
            return
        idx = len(self.struct_order)
        color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        self.structures[name] = {"color": color, "contours": {}}
        self.struct_order.append(name)
        self._refresh_struct_list()
        self.struct_lb.selection_clear(0, tk.END)
        self.struct_lb.selection_set(self.struct_order.index(name))
        self._on_struct_select()

    def _delete_structure(self):
        sel = self.struct_lb.curselection()
        if not sel:
            return
        name = self.struct_order[sel[0]]
        if not messagebox.askyesno("Delete", f"Delete structure '{name}'?"):
            return
        del self.structures[name]
        self.struct_order.remove(name)
        if self.active_struct == name:
            self.active_struct = None
        self._dvh_cache.pop(name, None)
        self._refresh_struct_list()
        self._update_display()

    def _rename_structure(self):
        sel = self.struct_lb.curselection()
        if not sel:
            return
        old = self.struct_order[sel[0]]
        new = simpledialog.askstring("Rename", f"New name for '{old}':",
                                     initialvalue=old, parent=self.master)
        if not new or new.strip() == "" or new == old:
            return
        new = new.strip()
        if new in self.structures:
            messagebox.showwarning("Exists", f"'{new}' already exists.")
            return
        self.structures[new] = self.structures.pop(old)
        idx = self.struct_order.index(old)
        self.struct_order[idx] = new
        if self.active_struct == old:
            self.active_struct = new
        if old in self._dvh_cache:
            self._dvh_cache[new] = self._dvh_cache.pop(old)
        self._refresh_struct_list()
        self.struct_lb.selection_set(idx)
        self._on_struct_select()

    def _change_struct_color(self):
        if not self.active_struct:
            return
        init = self.structures[self.active_struct]["color"]
        result = colorchooser.askcolor(color=init, title="Choose structure color",
                                       parent=self.master)
        if result and result[1]:
            self.structures[self.active_struct]["color"] = result[1]
            self._refresh_struct_list()
            self._update_color_swatch()
            self._update_display()

    def _on_struct_select(self, event=None):
        sel = self.struct_lb.curselection()
        if not sel:
            return
        self.active_struct = self.struct_order[sel[0]]
        self._update_color_swatch()
        self._update_contour_info()

    def _refresh_struct_list(self):
        self.struct_lb.delete(0, tk.END)
        for name in self.struct_order:
            n_contours = sum(len(v) for v in self.structures[name]["contours"].values())
            self.struct_lb.insert(tk.END, f"  {name}  [{n_contours} poly]")

    def _update_color_swatch(self):
        self.color_canvas.delete("all")
        if self.active_struct:
            c = self.structures[self.active_struct]["color"]
            w = self.color_canvas.winfo_width() or 200
            self.color_canvas.configure(bg=c)
            self.color_canvas.create_text(
                w // 2, 9, text=f"  {self.active_struct}  (click to change color)",
                fill="white" if self._is_dark(c) else "black", font=("", 9))

    def _is_dark(self, hex_color):
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (0.299 * r + 0.587 * g + 0.114 * b) < 128

    def _update_contour_info(self):
        if not self.active_struct:
            self.contour_info_var.set("")
            return
        s = self.structures[self.active_struct]
        n_slices = len(s["contours"])
        n_poly   = sum(len(v) for v in s["contours"].values())
        self.contour_info_var.set(
            f"Active: {self.active_struct}  |  {n_slices} slices  {n_poly} polygons")

    # ══════════════════════════════════════════════════════════════════
    # Drawing / Contouring
    # ══════════════════════════════════════════════════════════════════

    def _toggle_draw_mode(self):
        self.draw_mode = self.draw_mode_var.get()
        lbl = "Draw mode (ON – click axial to add pts)" if self.draw_mode else "Draw mode (OFF)"
        self.draw_chk.configure(text=lbl)
        if not self.draw_mode:
            self._draw_pts.clear()
            self._draw_mouse_mm = None
        self._update_display()

    def _close_contour(self):
        if len(self._draw_pts) < 3:
            messagebox.showinfo("Contouring", "Need at least 3 points to close a polygon.")
            return
        if not self.active_struct:
            messagebox.showwarning("Contouring", "Select a structure first.")
            return
        z = self.cur_z
        contours = self.structures[self.active_struct]["contours"]
        contours.setdefault(z, []).append(list(self._draw_pts))
        self._draw_pts.clear()
        self._draw_mouse_mm = None
        self._dvh_cache.pop(self.active_struct, None)
        self._refresh_struct_list()
        self._update_contour_info()
        self._update_display()

    def _cancel_draw(self):
        self._draw_pts.clear()
        self._draw_mouse_mm = None
        self._update_display()

    def _clear_slice(self):
        if not self.active_struct:
            return
        z = self.cur_z
        self.structures[self.active_struct]["contours"].pop(z, None)
        self._dvh_cache.pop(self.active_struct, None)
        self._refresh_struct_list()
        self._update_contour_info()
        self._update_display()

    def _clear_all(self):
        if not self.active_struct:
            return
        if not messagebox.askyesno("Clear all",
                                   f"Clear all contours for '{self.active_struct}'?"):
            return
        self.structures[self.active_struct]["contours"].clear()
        self._dvh_cache.pop(self.active_struct, None)
        self._refresh_struct_list()
        self._update_contour_info()
        self._update_display()

    # ══════════════════════════════════════════════════════════════════
    # Navigation
    # ══════════════════════════════════════════════════════════════════

    def _step_z(self, d):
        if self.ct_volume is None: return
        self.cur_z = int(np.clip(self.cur_z + d, 0, self.ct_volume.shape[0] - 1))
        self.z_var.set(str(self.cur_z)); self._update_display()

    def _step_x(self, d):
        if self.ct_volume is None: return
        self.cur_x = int(np.clip(self.cur_x + d, 0, self.ct_volume.shape[2] - 1))
        self.x_var.set(str(self.cur_x)); self._update_display()

    def _step_y(self, d):
        if self.ct_volume is None: return
        self.cur_y = int(np.clip(self.cur_y + d, 0, self.ct_volume.shape[1] - 1))
        self.y_var.set(str(self.cur_y)); self._update_display()

    def _go_z(self):
        if self.ct_volume is None: return
        try: v = int(self.z_var.get())
        except ValueError: return
        self.cur_z = int(np.clip(v, 0, self.ct_volume.shape[0] - 1))
        self.z_var.set(str(self.cur_z)); self._update_display()

    def _go_x(self):
        if self.ct_volume is None: return
        try: v = int(self.x_var.get())
        except ValueError: return
        self.cur_x = int(np.clip(v, 0, self.ct_volume.shape[2] - 1))
        self.x_var.set(str(self.cur_x)); self._update_display()

    def _go_y(self):
        if self.ct_volume is None: return
        try: v = int(self.y_var.get())
        except ValueError: return
        self.cur_y = int(np.clip(v, 0, self.ct_volume.shape[1] - 1))
        self.y_var.set(str(self.cur_y)); self._update_display()

    # ══════════════════════════════════════════════════════════════════
    # Events
    # ══════════════════════════════════════════════════════════════════

    def _on_scroll(self, event):
        delta = 1 if event.button == "up" else -1
        if   event.inaxes == self.ax_ax:  self._step_z(delta)
        elif event.inaxes == self.ax_sag: self._step_x(delta)
        elif event.inaxes == self.ax_cor: self._step_y(delta)

    def _on_click(self, event):
        if self.ct_volume is None or event.xdata is None:
            return
        nz, ny, nx = self.ct_volume.shape
        col_mm = self.ct_meta["pixel_spacing"][1]
        row_mm = self.ct_meta["pixel_spacing"][0]
        st_mm  = self.ct_meta["slice_thickness"]

        if self.draw_mode and event.inaxes == self.ax_ax:
            x_mm = event.xdata; y_mm = event.ydata
            if len(self._draw_pts) >= 3:
                fx, fy = self._draw_pts[0]
                if np.hypot(x_mm - fx, y_mm - fy) < 3.0 * col_mm:
                    self._close_contour()
                    return
            self._draw_pts.append((x_mm, y_mm))
            self._update_display()
            return

        if event.inaxes == self.ax_ax:
            self.cur_x = int(np.clip(event.xdata / col_mm, 0, nx - 1))
            self.cur_y = int(np.clip(event.ydata / row_mm, 0, ny - 1))
            self.x_var.set(str(self.cur_x)); self.y_var.set(str(self.cur_y))
        elif event.inaxes == self.ax_sag:
            self.cur_y = int(np.clip(event.xdata / row_mm, 0, ny - 1))
            self.cur_z = int(np.clip(event.ydata / st_mm,  0, nz - 1))
            self.y_var.set(str(self.cur_y)); self.z_var.set(str(self.cur_z))
        elif event.inaxes == self.ax_cor:
            self.cur_x = int(np.clip(event.xdata / col_mm, 0, nx - 1))
            self.cur_z = int(np.clip(event.ydata / st_mm,  0, nz - 1))
            self.x_var.set(str(self.cur_x)); self.z_var.set(str(self.cur_z))
        else:
            return
        self._update_display()

    def _on_motion(self, event):
        if not self.draw_mode or event.inaxes != self.ax_ax:
            self._draw_mouse_mm = None
            return
        if event.xdata is not None:
            self._draw_mouse_mm = (event.xdata, event.ydata)
            if self._draw_pts:
                if self._motion_after is not None:
                    self.master.after_cancel(self._motion_after)
                self._motion_after = self.master.after(40, self._update_display)

    def _on_key(self, event):
        key = event.keysym
        if   key in ("Up",    "w"): self._step_z(1)
        elif key in ("Down",  "s"): self._step_z(-1)
        elif key in ("Right", "d"): self._step_x(1)
        elif key in ("Left",  "a"): self._step_x(-1)
        elif key == "Return" and self.draw_mode:
            self._close_contour()
        elif key == "Escape":
            self._cancel_draw()

    # ══════════════════════════════════════════════════════════════════
    # Rendering
    # ══════════════════════════════════════════════════════════════════

    def _update_display(self):
        if self._cbar is not None:
            try: self._cbar.remove()
            except Exception: pass
            self._cbar = None
        for ax in (self.ax_ax, self.ax_sag, self.ax_cor, self.ax_dvh):
            ax.clear()

        self.fig.patch.set_facecolor(BG)
        for _ax in (self.ax_ax, self.ax_sag, self.ax_cor, self.ax_dvh):
            _ax.set_facecolor(AX_BG)
            _ax.tick_params(colors=FG_DIM, labelsize=7)
            _ax.xaxis.label.set_color(FG_DIM)
            _ax.yaxis.label.set_color(FG_DIM)
            _ax.title.set_color(FG)
            for spine in _ax.spines.values():
                spine.set_color(BORDER)
                spine.set_linewidth(0.8)

        if self.ct_volume is None:
            for ax in (self.ax_ax, self.ax_sag, self.ax_cor):
                ax.axis("off")
            self.ax_ax.text(0.5, 0.5, "Load a CT DICOM folder to begin",
                            transform=self.ax_ax.transAxes,
                            ha="center", va="center", fontsize=13, color=FG_DIM)
            self.ax_dvh.axis("off")
            self.canvas.draw_idle()
            return

        nz, ny, nx = self.ct_volume.shape
        col_mm = self.ct_meta["pixel_spacing"][1]
        row_mm = self.ct_meta["pixel_spacing"][0]
        st_mm  = self.ct_meta["slice_thickness"]
        vmin   = self.win_level - self.win_width / 2.0
        vmax   = self.win_level + self.win_width / 2.0
        show_dose = self.dose_visible and self.dose_volume is not None

        # ── Axial ──────────────────────────────────────────────────────
        ax_sl  = self.ct_volume[self.cur_z, :, :]
        ax_ext = [0, nx * col_mm, ny * row_mm, 0]
        self.ax_ax.imshow(ax_sl, cmap="gray", vmin=vmin, vmax=vmax,
                          extent=ax_ext, origin="upper", aspect="equal")
        if show_dose:
            self.ax_ax.imshow(self._dose_rgba(self.dose_volume[self.cur_z]),
                              extent=ax_ext, origin="upper", aspect="equal")
        self.ax_ax.axvline(self.cur_x * col_mm, color=XHAIR, lw=0.8, alpha=0.8)
        self.ax_ax.axhline(self.cur_y * row_mm, color=XHAIR, lw=0.8, alpha=0.8)

        for name in self.struct_order:
            st = self.structures[name]
            c  = st["color"]
            for pts in st["contours"].get(self.cur_z, []):
                if len(pts) >= 2:
                    xs = [p[0] for p in pts] + [pts[0][0]]
                    ys = [p[1] for p in pts] + [pts[0][1]]
                    self.ax_ax.plot(xs, ys, "-", color=c, lw=1.5, alpha=0.9)
                    self.ax_ax.fill(xs, ys, color=c, alpha=0.15)
                    cx = np.mean([p[0] for p in pts])
                    cy = np.mean([p[1] for p in pts])
                    self.ax_ax.text(cx, cy, name, fontsize=6, color=c,
                                    ha="center", va="center")

        if self.draw_mode and self._draw_pts:
            xs = [p[0] for p in self._draw_pts]
            ys = [p[1] for p in self._draw_pts]
            self.ax_ax.plot(xs, ys, "w--", lw=1.5, alpha=0.9)
            self.ax_ax.plot(xs, ys, "wo",  ms=4,  alpha=0.9)
            if self._draw_mouse_mm:
                mx, my = self._draw_mouse_mm
                self.ax_ax.plot([xs[-1], mx], [ys[-1], my], "w--", lw=1, alpha=0.5)

        mode_txt = "  ● DRAW" if self.draw_mode else ""
        self.ax_ax.set_title(
            f"AXIAL   Z={self.cur_z}  ({self.cur_z*st_mm:.1f} mm){mode_txt}",
            fontsize=8, loc="left", color=FG, pad=4)
        self.ax_ax.set_xlabel("X (mm)", fontsize=7)
        self.ax_ax.set_ylabel("Y (mm)", fontsize=7)

        # ── Sagittal ───────────────────────────────────────────────────
        sag_sl  = self.ct_volume[:, :, self.cur_x]
        sag_ext = [0, ny * row_mm, nz * st_mm, 0]
        self.ax_sag.imshow(sag_sl, cmap="gray", vmin=vmin, vmax=vmax,
                           extent=sag_ext, origin="upper", aspect="equal")
        if show_dose:
            self.ax_sag.imshow(self._dose_rgba(self.dose_volume[:, :, self.cur_x]),
                               extent=sag_ext, origin="upper", aspect="equal")
        self.ax_sag.axvline(self.cur_y * row_mm, color=XHAIR, lw=0.8, alpha=0.8)
        self.ax_sag.axhline(self.cur_z * st_mm,  color=XHAIR, lw=0.8, alpha=0.8)

        x_probe = (self.cur_x + 0.5) * col_mm
        for name in self.struct_order:
            st_s = self.structures[name]; c = st_s["color"]
            for z_idx, poly_list in st_s["contours"].items():
                z_mm = (z_idx + 0.5) * st_mm
                for pts in poly_list:
                    ys_cross = polygon_x_intersections(pts, x_probe)
                    for i in range(0, len(ys_cross) - 1, 2):
                        self.ax_sag.plot([ys_cross[i], ys_cross[i + 1]],
                                         [z_mm, z_mm], "-", color=c, lw=2.0, alpha=0.85)

        self.ax_sag.set_title(
            f"SAGITTAL   X={self.cur_x}  ({self.cur_x*col_mm:.1f} mm)",
            fontsize=8, loc="left", color=FG, pad=4)
        self.ax_sag.set_xlabel("Y (mm)", fontsize=7)
        self.ax_sag.set_ylabel("Z (mm)", fontsize=7)

        # ── Coronal ────────────────────────────────────────────────────
        cor_sl  = self.ct_volume[:, self.cur_y, :]
        cor_ext = [0, nx * col_mm, nz * st_mm, 0]
        self.ax_cor.imshow(cor_sl, cmap="gray", vmin=vmin, vmax=vmax,
                           extent=cor_ext, origin="upper", aspect="equal")
        if show_dose:
            self.ax_cor.imshow(self._dose_rgba(self.dose_volume[:, self.cur_y, :]),
                               extent=cor_ext, origin="upper", aspect="equal")
        self.ax_cor.axvline(self.cur_x * col_mm, color=XHAIR, lw=0.8, alpha=0.8)
        self.ax_cor.axhline(self.cur_z * st_mm,  color=XHAIR, lw=0.8, alpha=0.8)

        y_probe = (self.cur_y + 0.5) * row_mm
        for name in self.struct_order:
            st_c = self.structures[name]; c = st_c["color"]
            for z_idx, poly_list in st_c["contours"].items():
                z_mm = (z_idx + 0.5) * st_mm
                for pts in poly_list:
                    xs_cross = polygon_y_intersections(pts, y_probe)
                    for i in range(0, len(xs_cross) - 1, 2):
                        self.ax_cor.plot([xs_cross[i], xs_cross[i + 1]],
                                         [z_mm, z_mm], "-", color=c, lw=2.0, alpha=0.85)

        self.ax_cor.set_title(
            f"CORONAL   Y={self.cur_y}  ({self.cur_y*row_mm:.1f} mm)",
            fontsize=8, loc="left", color=FG, pad=4)
        self.ax_cor.set_xlabel("X (mm)", fontsize=7)
        self.ax_cor.set_ylabel("Z (mm)", fontsize=7)

        # ── Dose colorbar ─────────────────────────────────────────────
        if show_dose:
            dmax = float(np.max(self.dose_volume))
            if dmax > 0:
                sm = plt.cm.ScalarMappable(cmap="jet",
                                           norm=plt.Normalize(vmin=0, vmax=dmax))
                sm.set_array([])
                self._cbar = self.fig.colorbar(sm, ax=self.ax_sag, fraction=0.04,
                                               pad=0.02, label="Dose (Gy)")
                self._cbar.ax.yaxis.set_tick_params(color=FG_DIM, labelcolor=FG_DIM)
                self._cbar.outline.set_edgecolor(BORDER)
                self._cbar.set_label("Dose (Gy)", color=FG_DIM)

        # ── DVH panel ─────────────────────────────────────────────────
        self._draw_dvh_panel()

        # ── Position readout ──────────────────────────────────────────
        hu = self.ct_volume[self.cur_z, self.cur_y, self.cur_x]
        self.pos_var.set(
            f"X={self.cur_x} ({self.cur_x*col_mm:.1f} mm)  "
            f"Y={self.cur_y} ({self.cur_y*row_mm:.1f} mm)  "
            f"Z={self.cur_z} ({self.cur_z*st_mm:.1f} mm)\n"
            f"HU = {hu:.1f}")

        self.canvas.draw_idle()

    def _invalidate_dvh_cache(self):
        self._dvh_cache.clear()

    def _draw_dvh_panel(self):
        ax = self.ax_dvh
        has_dose    = self.dose_volume is not None
        has_structs = any(len(s["contours"]) > 0 for s in self.structures.values())

        if not has_dose or not has_structs:
            ax.axis("off")
            ax.set_facecolor(AX_BG)
            ax.text(0.5, 0.5, "DVH\n\nLoad dose + draw\ncontours to compute.",
                    transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color=FG_DIM)
            return

        any_plotted = False
        for name in self.struct_order:
            st = self.structures[name]
            if not st["contours"]:
                continue
            if name not in self._dvh_cache:
                self._dvh_cache[name] = compute_dvh(self.dose_volume, st["contours"], self.ct_meta)
            doses, vols, stats = self._dvh_cache[name]
            ax.plot(doses, vols, color=st["color"], lw=2, label=name)
            any_plotted = True

        if any_plotted:
            ax.set_xlabel("Dose (Gy)", fontsize=8, color=FG_DIM)
            ax.set_ylabel("Volume (%)", fontsize=8, color=FG_DIM)
            ax.set_title("DVH", fontsize=8, loc="left", color=FG, pad=4)
            ax.set_xlim(left=0)
            ax.set_ylim(0, 105)
            ax.grid(True, color=BORDER, alpha=0.5, linewidth=0.5)
            ax.legend(fontsize=7, loc="upper right",
                      facecolor=BG2, edgecolor=BORDER, labelcolor=FG)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "No contoured structures with dose.",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color=FG_DIM)

    # ══════════════════════════════════════════════════════════════════
    # Save / Load / Export
    # ══════════════════════════════════════════════════════════════════

    def _save_contours(self):
        if not self.structures:
            messagebox.showinfo("Nothing to save", "No structures defined.")
            return
        path = filedialog.asksaveasfilename(
            title="Save contours", defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        data = {
            "structures": {
                name: {
                    "color": st["color"],
                    "contours": {str(k): v for k, v in st["contours"].items()},
                }
                for name, st in self.structures.items()
            },
            "struct_order": self.struct_order,
        }
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        messagebox.showinfo("Saved", f"Contours saved to:\n{path}")

    def _load_contours(self):
        path = filedialog.askopenfilename(
            title="Load contours", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load:\n{e}")
            return
        self.structures.clear()
        self.struct_order.clear()
        for name, st in data["structures"].items():
            self.structures[name] = {
                "color": st["color"],
                "contours": {int(k): v for k, v in st["contours"].items()},
            }
        self.struct_order = data.get("struct_order", list(self.structures.keys()))
        self.active_struct = self.struct_order[0] if self.struct_order else None
        self._invalidate_dvh_cache()
        self._refresh_struct_list()
        self._update_contour_info()
        self._update_display()

    def _export_view(self):
        if self.ct_volume is None:
            messagebox.showwarning("Warning", "Load a CT first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export figure", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tiff"), ("All", "*.*")])
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Exported", f"Saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{e}")

    # ══════════════════════════════════════════════════════════════════
    # TOPAS
    # ══════════════════════════════════════════════════════════════════

    def _copy_ct_path(self):
        folder = self.ct_folder or ""
        if not folder:
            messagebox.showinfo("No CT", "Load a CT folder first.")
            return
        self.master.clipboard_clear()
        self.master.clipboard_append(folder)
        self.topas_status_var.set("CT path copied to clipboard.")

    def _pick_topas_macro(self):
        path = filedialog.askopenfilename(
            title="Select TOPAS macro",
            filetypes=[("Text", "*.txt"), ("TOPAS", "*.topas"), ("All", "*.*")])
        if path:
            self._topas_macro = path
            self.topas_macro_var.set(os.path.basename(path))

    def _run_topas(self):
        if not self._topas_macro:
            messagebox.showwarning("TOPAS", "Select a TOPAS macro file first.")
            return
        topas_exe = "topas"
        self.topas_status_var.set("Running TOPAS…")
        self.master.update_idletasks()
        try:
            result = subprocess.run(
                [topas_exe, self._topas_macro],
                capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                self.topas_status_var.set("TOPAS finished successfully.")
                folder = os.path.dirname(self._topas_macro)
                headers = glob.glob(os.path.join(folder, "*.header"))
                if headers:
                    if messagebox.askyesno("TOPAS",
                                           f"Found dose header:\n{headers[0]}\nLoad it now?"):
                        try:
                            vol, dmeta = load_topas_dose(headers[0])
                            self._set_dose(vol, dmeta, os.path.basename(headers[0]))
                        except Exception as ex:
                            messagebox.showerror("Error", f"Could not load dose:\n{ex}")
            else:
                self.topas_status_var.set(f"TOPAS error (code {result.returncode}).")
                messagebox.showerror("TOPAS error",
                                     f"stdout:\n{result.stdout[-500:]}\n\n"
                                     f"stderr:\n{result.stderr[-500:]}")
        except FileNotFoundError:
            self.topas_status_var.set("'topas' not found on PATH.")
            messagebox.showerror("TOPAS not found",
                                 "Could not find 'topas' executable.\n"
                                 "Make sure TOPAS is installed and on your PATH.")
        except subprocess.TimeoutExpired:
            self.topas_status_var.set("TOPAS timed out (>1 hr).")


# ═══════════════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    root.geometry("1400x900")
    app = TreatmentPlanner(root)
    root.mainloop()


if __name__ == "__main__":
    main()

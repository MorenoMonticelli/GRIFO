
"""
GRIFO.py - Desktop GUI for transit photometry (PySide6 + pyqtgraph).

Run with:
  python GRIFO.py

Dependencies:
  pip install PySide6 pyqtgraph astropy photutils scipy matplotlib numpy emcee batman-package corner
"""

import os
import json
import math
import time
import sys
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
import matplotlib.pyplot as plt

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from astropy.io import fits
from astropy.time import Time
from astropy import constants as const

from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.ndimage import shift as ndi_shift
from scipy.ndimage import rotate as ndi_rotate


# Optional
try:
    import emcee
    HAS_EMCEE = True
except Exception:
    HAS_EMCEE = False

try:
    import batman
    HAS_BATMAN = True
except Exception:
    HAS_BATMAN = False

try:
    import corner
    HAS_CORNER = True
except Exception:
    HAS_CORNER = False


try:
    RSUN_TO_AU = float((const.R_sun / const.au).value)
    RSUN_TO_RJUP = float((const.R_sun / const.R_jup).value)
    RSUN_TO_REARTH = float((const.R_sun / const.R_earth).value)
except Exception:
    # Fallback constants if astropy constants are unavailable.
    RSUN_TO_AU = 0.004650467261
    RSUN_TO_RJUP = 9.735
    RSUN_TO_REARTH = 109.1


# -------------------------
# Utility helpers
# -------------------------

def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def get_airmass_from_header(hdr):
    if hdr is None:
        return np.nan
    for k in ["AIRMASS", "AMSTART", "AMEND", "SECZ", "secz", "ZD", "ZENITH"]:
        if k in hdr:
            return _safe_float(hdr.get(k), np.nan)
    return np.nan


def get_time_jd_from_header(hdr):
    """
    Returns JD (float) or np.nan if missing.
    Tries BJD/JD-like keys first, then MJD-OBS, then DATE-OBS.
    """
    if hdr is None:
        return np.nan

    for k in ["BJD_TDB", "BJD-TDB", "BJDTDB", "BJD_TDB_MID", "BJD", "BJD-OBS", "HJD", "JD"]:
        if k in hdr:
            return _safe_float(hdr[k], np.nan)

    if "MJD-OBS" in hdr:
        mjd = _safe_float(hdr["MJD-OBS"], np.nan)
        if np.isfinite(mjd):
            return mjd + 2400000.5

    if "DATE-OBS" in hdr:
        try:
            return float(Time(hdr["DATE-OBS"]).jd)
        except Exception:
            return np.nan

    return np.nan


def get_exptime_from_header(hdr):
    if hdr is None:
        return np.nan
    for k in ["EXPTIME", "EXPOSURE", "ITIME", "EXP_TIME"]:
        if k in hdr:
            return _safe_float(hdr.get(k), np.nan)
    return np.nan


def decimate_2x2(img):
    """
    2x2 block-average binning (better photometric behavior than pixel dropping).
    """
    img = np.asarray(img)
    if not np.issubdtype(img.dtype, np.floating):
        img = img.astype(np.float32, copy=False)
    ny, nx = img.shape
    ny2 = (ny // 2) * 2
    nx2 = (nx // 2) * 2
    if ny2 == 0 or nx2 == 0:
        return img.copy()
    core = img[:ny2, :nx2]
    quarter = core.dtype.type(0.25)
    return quarter * (
        core[0::2, 0::2] + core[1::2, 0::2] +
        core[0::2, 1::2] + core[1::2, 1::2]
    )


def weighted_median(values, weights):
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if np.sum(m) == 0:
        return np.nan
    v = v[m]
    w = w[m]
    s = np.argsort(v)
    v = v[s]
    w = w[s]
    cw = np.cumsum(w)
    cutoff = 0.5 * np.sum(w)
    return float(v[np.searchsorted(cw, cutoff, side="left")])


def robust_sigma_from_mad(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad > 0:
        return float(1.4826 * mad)
    return float(np.std(v)) if v.size > 1 else 0.0


def time_axis_diagnostics(images):
    jd = np.asarray([im.get("jd", np.nan) for im in images], float)
    finite = np.isfinite(jd)
    jf = jd[finite]
    n_missing = int(np.sum(~finite))
    n_dup = int(max(0, jf.size - np.unique(np.round(jf, 10)).size))
    n_nonmono = int(np.sum(np.diff(jf) < 0)) if jf.size > 1 else 0

    has_bjd_like = int(np.sum([bool(im.get("has_bjd_like", False)) for im in images]))

    return {
        "n_total": int(jd.size),
        "n_missing_jd": n_missing,
        "n_duplicate_jd": n_dup,
        "n_non_monotonic": n_nonmono,
        "n_bjd_like": int(has_bjd_like),
    }

def load_fits_cube_from_paths(file_paths, bin2x2: bool = True, dtype: str = "float32"):
    """
    Load FITS files from local paths into cube + metadata list.
    file_paths: list[str]
    Returns: cube, images
    """
    if file_paths is None or len(file_paths) == 0:
        raise FileNotFoundError("No FITS files selected.")

    paths = sorted([str(p) for p in file_paths])
    cube_list = []
    images = []

    out_dtype = np.float32 if dtype == "float32" else np.float64

    for fp in paths:
        with fits.open(fp, memmap=False) as hdul:
            data = hdul[0].data
            hdr = hdul[0].header

        data = np.asarray(data)
        data = np.squeeze(data)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D FITS image in '{fp}', found shape={data.shape}.")
        data = data.astype(out_dtype, copy=False)

        if bin2x2:
            data = decimate_2x2(data)

        filt = hdr.get("FILTER", "NA")
        jd = get_time_jd_from_header(hdr)
        am = get_airmass_from_header(hdr)
        exptime = get_exptime_from_header(hdr)
        has_bjd_like = any(k in hdr for k in ["BJD_TDB", "BJD-TDB", "BJDTDB", "BJD_TDB_MID"])

        cube_list.append(data)
        images.append(
            {
                "file": fp,
                "filter": filt,
                "jd": jd,
                "airmass": am,
                "exptime": exptime,
                "has_bjd_like": bool(has_bjd_like),
            }
        )

    cube = np.stack(cube_list, axis=0)
    return cube, images


def percentile_vmin_vmax(img, pmin=5.0, pmax=99.5):
    vmin = np.nanpercentile(img, pmin)
    vmax = np.nanpercentile(img, pmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(img)), float(np.nanmax(img))
    return vmin, vmax


def centroid_2d(img, x0, y0, half_size=10):
    """
    Flux-weighted centroid inside a small window.
    Returns (xc, yc). If window invalid, returns original.
    """
    img = np.asarray(img)
    if not np.issubdtype(img.dtype, np.floating):
        img = img.astype(np.float32, copy=False)
    ny, nx = img.shape
    x0i = int(round(x0)); y0i = int(round(y0))

    x1 = max(0, x0i-half_size); x2 = min(nx, x0i+half_size+1)
    y1 = max(0, y0i-half_size); y2 = min(ny, y0i+half_size+1)

    cut = img[y1:y2, x1:x2]
    if cut.size == 0:
        return float(x0), float(y0)

    # subtract local background (robust)
    med = np.nanmedian(cut)
    w = cut - med
    w[w < 0] = 0

    tot = np.nansum(w)
    if not np.isfinite(tot) or tot <= 0:
        return float(x0), float(y0)

    yy, xx = np.indices(cut.shape)
    xc = np.nansum(xx * w) / tot + x1
    yc = np.nansum(yy * w) / tot + y1
    return float(xc), float(yc)

def _circular_subpixel_fractions(xx, yy, x0, y0, r_ap, r_in, r_out, subpixels=5):
    sub = max(1, int(subpixels))
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    xsp = xx[:, :, None, None] + offs[None, None, :, None]
    ysp = yy[:, :, None, None] + offs[None, None, None, :]
    rr = np.sqrt((xsp - x0) ** 2 + (ysp - y0) ** 2)
    ap_frac = np.mean(rr <= r_ap, axis=(2, 3))
    an_frac = np.mean((rr >= r_in) & (rr <= r_out), axis=(2, 3))
    return ap_frac, an_frac


def _fit_background_plane(xx, yy, zz, ww, sigma_clip=3.0, max_iter=3):
    xx = np.asarray(xx, float).ravel()
    yy = np.asarray(yy, float).ravel()
    zz = np.asarray(zz, float).ravel()
    ww = np.asarray(ww, float).ravel()

    m = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(zz) & np.isfinite(ww) & (ww > 0)
    idx = np.where(m)[0]
    if idx.size < 3:
        return None, np.nan

    coeffs = None
    for _ in range(max_iter):
        xk = xx[idx]
        yk = yy[idx]
        zk = zz[idx]
        wk = np.sqrt(np.clip(ww[idx], 1e-9, None))
        X = np.column_stack([xk, yk, np.ones_like(xk)])
        Aw = X * wk[:, None]
        bw = zk * wk
        coeffs, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
        model = X @ coeffs
        resid = zk - model
        sigma = robust_sigma_from_mad(resid)
        if (sigma_clip is None) or (not np.isfinite(sigma)) or sigma <= 0:
            break
        keep = np.abs(resid) <= float(sigma_clip) * sigma
        if np.sum(keep) < 3 or np.all(keep):
            break
        idx = idx[keep]

    if coeffs is None:
        return None, np.nan

    sigma = robust_sigma_from_mad(zz[idx] - (xx[idx] * coeffs[0] + yy[idx] * coeffs[1] + coeffs[2]))
    return coeffs, sigma


def aperture_photometry_fast(
    img, x, y, r_ap, r_in, r_out,
    bkg_stat="median",
    bkg_sigma_clip=3.0,
    gain_e_per_adu=1.0,
    read_noise_e=0.0,
    bkg_model="annulus",
    subpixels=5,
    saturation_level=None,
):
    """
    Returns ALWAYS 4 values:
      flux_adu, sigma_flux_adu, bkg_per_pix_adu, extra_dict
    """
    img = np.asarray(img)
    if not np.issubdtype(img.dtype, np.floating):
        img = img.astype(np.float32, copy=False)
    ny, nx = img.shape

    r_ap = float(r_ap)
    r_in = float(r_in)
    r_out = float(r_out)
    if not (r_ap > 0 and r_in > r_ap and r_out > r_in):
        return np.nan, np.nan, np.nan, {"n_ap": 0, "n_an": 0, "reason": "invalid_radii"}

    rmax = float(r_out)
    pad = int(np.ceil(rmax + 2))
    x0 = int(np.floor(x)) - pad
    x1 = int(np.floor(x)) + pad + 1
    y0 = int(np.floor(y)) - pad
    y1 = int(np.floor(y)) + pad + 1
    x0c = max(0, x0)
    x1c = min(nx, x1)
    y0c = max(0, y0)
    y1c = min(ny, y1)

    cut = img[y0c:y1c, x0c:x1c]
    if cut.size == 0:
        return np.nan, np.nan, np.nan, {"n_ap": 0, "n_an": 0, "reason": "empty_cutout"}

    yy, xx = np.indices(cut.shape)
    xx = xx + x0c
    yy = yy + y0c

    ap_frac, an_frac = _circular_subpixel_fractions(xx, yy, x, y, r_ap, r_in, r_out, subpixels=subpixels)
    valid = np.isfinite(cut)
    ap_frac = np.where(valid, ap_frac, 0.0)
    an_frac = np.where(valid, an_frac, 0.0)

    n_ap = float(np.sum(ap_frac))
    n_an = float(np.sum(an_frac))
    if n_ap <= 0:
        return np.nan, np.nan, np.nan, {"n_ap": 0, "n_an": int(round(n_an)), "reason": "empty_aperture"}

    ap_sum = float(np.sum(cut * ap_frac))

    an_sel = an_frac > 0
    ann_vals = cut[an_sel]
    ann_w = an_frac[an_sel]
    ann_x = xx[an_sel]
    ann_y = yy[an_sel]

    bkg_per_pix = 0.0
    bkg_std = 0.0
    bkg_ap_sum = 0.0
    bkg_model = str(bkg_model).strip().lower()
    bkg_stat = str(bkg_stat).strip().lower()

    if ann_vals.size > 0:
        m = np.isfinite(ann_vals) & np.isfinite(ann_w) & (ann_w > 0)
        v = ann_vals[m]
        w = ann_w[m]
        ax = ann_x[m]
        ay = ann_y[m]

        if v.size > 0:
            if bkg_sigma_clip is not None and v.size > 10:
                center = weighted_median(v, w) if bkg_stat == "median" else float(np.average(v, weights=w))
                sig = robust_sigma_from_mad(v - center)
                if np.isfinite(sig) and sig > 0:
                    keep = np.abs(v - center) <= float(bkg_sigma_clip) * sig
                    if np.any(keep):
                        v = v[keep]
                        w = w[keep]
                        ax = ax[keep]
                        ay = ay[keep]

            if bkg_model == "plane" and v.size >= 8:
                coeffs, plane_sigma = _fit_background_plane(ax, ay, v, w, sigma_clip=bkg_sigma_clip, max_iter=3)
                if coeffs is not None:
                    plane = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
                    bkg_ap_sum = float(np.sum(plane * ap_frac))
                    bkg_per_pix = float(bkg_ap_sum / max(n_ap, 1e-12))
                    bkg_std = float(plane_sigma) if np.isfinite(plane_sigma) else 0.0
                else:
                    bkg_model = "annulus"

            if bkg_model != "plane":
                bkg_per_pix = float(np.average(v, weights=w)) if bkg_stat == "mean" else float(weighted_median(v, w))
                resid = v - bkg_per_pix
                bkg_std = float(np.sqrt(np.average(resid**2, weights=w))) if v.size > 1 else 0.0
                bkg_ap_sum = float(bkg_per_pix * n_ap)

    flux = ap_sum - bkg_ap_sum

    gain = float(gain_e_per_adu) if gain_e_per_adu and gain_e_per_adu > 0 else 1.0
    rn_e = float(read_noise_e) if read_noise_e and read_noise_e > 0 else 0.0

    src_e = max(ap_sum * gain, 0.0)
    bkg_in_ap_e = max(bkg_ap_sum * gain, 0.0)
    var_bkg_pix_e2 = (bkg_std * gain) ** 2

    var_poisson_e = src_e + bkg_in_ap_e
    var_read_e = n_ap * (rn_e ** 2)
    var_bkgsub_e = (n_ap ** 2) * (var_bkg_pix_e2 / max(n_an, 1e-12))
    var_total_e = var_poisson_e + var_read_e + var_bkgsub_e
    sigma_flux = np.sqrt(max(var_total_e, 0.0)) / gain

    peak = float(np.nanmax(cut[ap_frac > 0])) if np.any(ap_frac > 0) else np.nan
    is_saturated = bool(np.isfinite(saturation_level) and np.isfinite(peak) and (peak >= float(saturation_level))) if saturation_level is not None else False
    near_edge = bool(x0c == 0 or y0c == 0 or x1c == nx or y1c == ny)

    extra = {
        "n_ap": int(round(n_ap)),
        "n_an": int(round(n_an)),
        "ap_sum": float(ap_sum),
        "bkg_std": float(bkg_std),
        "var_total_e": float(var_total_e),
        "peak": peak,
        "is_saturated": is_saturated,
        "near_edge": near_edge,
    }
    return flux, sigma_flux, bkg_per_pix, extra


def combine_comps(values, errors, mode="median", sigma_clip=None):
    values = np.asarray(values, float)
    errors = np.asarray(errors, float)

    m = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    if np.sum(m) == 0:
        return np.nan, np.nan

    v = values[m]
    e = errors[m]
    n = v.size

    mode = str(mode).strip().lower()

    if sigma_clip is not None and n >= 3 and sigma_clip > 0:
        med = np.median(v)
        sig = robust_sigma_from_mad(v - med)
        if np.isfinite(sig) and sig > 0:
            keep = np.abs(v - med) <= float(sigma_clip) * sig
            if np.any(keep):
                v = v[keep]
                e = e[keep]
                n = v.size
                if n == 0:
                    return np.nan, np.nan

    if mode == "sum":
        vref = float(np.sum(v))
        eref = float(np.sqrt(np.sum(e**2)))
        return vref, eref

    if mode in ("weighted", "wmean", "ivw"):
        w = 1.0 / (e**2)
        vref = float(np.sum(w * v) / np.sum(w))
        eref = float(np.sqrt(1.0 / np.sum(w)))
        return vref, eref

    if mode == "mean":
        vref = float(np.mean(v))
        eref = float(np.sqrt(np.sum(e**2)) / max(n, 1))
        return vref, eref

    if mode == "median":
        vref = float(np.median(v))
        if n == 1:
            eref = float(e[0])
        else:
            # rough teaching-friendly error on the median:
            # (use typical error / sqrt(n); 1.253 is mean->median factor for Gaussian)
            eref = float(1.253 * np.median(e) / np.sqrt(n))
        return vref, eref

    # Fallback: weighted mean
    w = 1.0 / (e**2)
    vref = float(np.sum(w * v) / np.sum(w))
    eref = float(np.sqrt(1.0 / np.sum(w)))
    return vref, eref


# -------------------------
# Alignment (photutils translation)
# -------------------------

def detect_stars_daofinder(img, fwhm=3.0, threshold_sigma=5.0, brightest=200):
    img = np.asarray(img)
    if not np.issubdtype(img.dtype, np.floating):
        img = img.astype(np.float32, copy=False)
    else:
        img = img.astype(np.float32, copy=False)
    mean, med, std = sigma_clipped_stats(img, sigma=3.0)
    finder = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
    tbl = finder(img - med)

    if tbl is None or len(tbl) == 0:
        return np.empty((0, 2), dtype=float)

    tbl.sort("flux")
    tbl = tbl[::-1][:brightest]

    x = np.array(tbl["xcentroid"], dtype=float)
    y = np.array(tbl["ycentroid"], dtype=float)
    return np.column_stack([x, y])


def robust_translation_from_points(ref_xy, xy, binsize=2.0, max_shift=200.0):
    if ref_xy.shape[0] == 0 or xy.shape[0] == 0:
        return np.nan, np.nan

    dx = (ref_xy[:, 0:1] - xy[None, :, 0]).ravel()
    dy = (ref_xy[:, 1:2] - xy[None, :, 1]).ravel()

    m = (np.abs(dx) <= max_shift) & (np.abs(dy) <= max_shift)
    dx = dx[m]; dy = dy[m]
    if dx.size == 0:
        return np.nan, np.nan

    xedges = np.arange(-max_shift, max_shift + binsize, binsize)
    yedges = np.arange(-max_shift, max_shift + binsize, binsize)
    H, xe, ye = np.histogram2d(dx, dy, bins=[xedges, yedges])

    i_dx, i_dy = np.unravel_index(np.argmax(H), H.shape)

    dx_min, dx_max = xe[i_dx], xe[i_dx + 1]
    dy_min, dy_max = ye[i_dy], ye[i_dy + 1]

    sel = (dx >= dx_min) & (dx < dx_max) & (dy >= dy_min) & (dy < dy_max)
    if np.sum(sel) < 10:
        return float(np.median(dx)), float(np.median(dy))

    return float(np.median(dx[sel])), float(np.median(dy[sel]))


def principal_axis_angle_deg(xy):
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or xy.shape[0] < 3:
        return 0.0
    c = np.cov(xy.T)
    if c.shape != (2, 2) or not np.all(np.isfinite(c)):
        return 0.0
    theta = 0.5 * np.arctan2(2.0 * c[0, 1], c[0, 0] - c[1, 1])
    return float(np.degrees(theta))


def align_cube_translation(cube, ref_index=0, fwhm=3.0, thr_sigma=5.0, brightest=150,
                           binsize=2.0, max_shift=200.0, order=3, progress_cb=None,
                           model="translation", max_rotation_deg=5.0, max_frames=None):
    ref = cube[ref_index]
    ref_xy = detect_stars_daofinder(ref, fwhm=fwhm, threshold_sigma=thr_sigma, brightest=brightest)
    ref_angle = principal_axis_angle_deg(ref_xy)

    n = cube.shape[0]
    aligned = np.empty_like(cube, dtype=cube.dtype)
    shifts_yx = np.zeros((n, 2), dtype=np.float32)
    rotations_deg = np.zeros(n, dtype=np.float32)

    for i in range(n):
        if (max_frames is not None) and (i >= int(max_frames)):
            aligned[i] = cube[i]
            shifts_yx[i] = (np.nan, np.nan)
            rotations_deg[i] = 0.0
            if progress_cb is not None:
                progress_cb(i + 1, n, np.nan, np.nan, 0.0)
            continue

        img_i = cube[i]
        rot_deg = 0.0

        if str(model).lower().startswith("translation+rotation"):
            xy_i = detect_stars_daofinder(img_i, fwhm=fwhm, threshold_sigma=thr_sigma, brightest=brightest)
            ang_i = principal_axis_angle_deg(xy_i)
            rot_deg = np.clip(ref_angle - ang_i, -abs(max_rotation_deg), abs(max_rotation_deg))
            img_i = ndi_rotate(
                img_i, angle=rot_deg, reshape=False, order=order,
                mode="constant", cval=0.0, prefilter=True
            )

        xy = detect_stars_daofinder(img_i, fwhm=fwhm, threshold_sigma=thr_sigma, brightest=brightest)
        dx, dy = robust_translation_from_points(ref_xy, xy, binsize=binsize, max_shift=max_shift)

        if not np.isfinite(dx) or not np.isfinite(dy):
            dx, dy = 0.0, 0.0

        rotations_deg[i] = float(rot_deg)
        shifts_yx[i] = (dy, dx)
        aligned[i] = ndi_shift(img_i, shift=(dy, dx), order=order, mode="constant", cval=0.0, prefilter=True)

        if progress_cb is not None:
            progress_cb(i + 1, n, dy, dx, rot_deg)

    return aligned, shifts_yx, rotations_deg


# -------------------------
# Detrending (time/airmass poly)
# -------------------------

def poly_design_matrix(x, degree):
    x = np.asarray(x, dtype=float)
    return np.vstack([x**d for d in range(degree + 1)]).T


def weighted_polyfit(x, y, yerr, degree):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    yerr = np.asarray(yerr, float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    if np.sum(m) < (degree + 1):
        return None, np.full_like(y, np.nan), m

    X = poly_design_matrix(x[m], degree)
    w = 1.0 / (yerr[m] ** 2)

    XT_W = X.T * w
    A = XT_W @ X
    b = XT_W @ y[m]
    try:
        coeffs = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    if not np.all(np.isfinite(coeffs)):
        return None, np.full_like(y, np.nan), m

    y_fit = np.full_like(y, np.nan)
    y_fit[m] = (X @ coeffs)
    return coeffs, y_fit, m


def detrend_flux(rel, srel, xtrend, degree=1, center_x=True, fit_mask=None):
    rel = np.asarray(rel, float)
    srel = np.asarray(srel, float)
    x = np.asarray(xtrend, float)

    if center_x:
        x = x - np.nanmedian(x)

    if fit_mask is None:
        fit_mask = np.ones_like(rel, dtype=bool)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    fit_mask = fit_mask & np.isfinite(x) & np.isfinite(rel) & np.isfinite(srel) & (srel > 0)

    if np.sum(fit_mask) < (degree + 1):
        return rel, srel, np.ones_like(rel)

    X_fit = poly_design_matrix(x[fit_mask], degree)
    w_fit = 1.0 / (srel[fit_mask] ** 2)
    A = (X_fit.T * w_fit) @ X_fit
    b = (X_fit.T * w_fit) @ rel[fit_mask]
    try:
        coeffs = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    if not np.all(np.isfinite(coeffs)):
        return rel, srel, np.ones_like(rel)

    trend = np.full_like(rel, np.nan, dtype=float)
    x_eval = np.isfinite(x)
    if np.any(x_eval):
        trend[x_eval] = poly_design_matrix(x[x_eval], degree) @ coeffs

    rel_d = rel / trend
    srel_d = srel / np.maximum(np.abs(trend), 1e-12)
    return rel_d, srel_d, trend


# -------------------------
# MCMC (batman)
# -------------------------

def batman_flux_model(t_rel, t0_rel, rp, a, inc, baseline, P, ecc, w, u1, u2):
    params = batman.TransitParams()
    params.t0 = t0_rel
    params.per = P
    params.rp = rp
    params.a = a
    params.inc = inc
    params.ecc = ecc
    params.w = w
    params.limb_dark = "quadratic"
    params.u = [u1, u2]
    m = batman.TransitModel(params, t_rel)
    return baseline * m.light_curve(params)


def run_batman_mcmc(t, y, yerr, guesses, fixed, walkers=64, burn=1000, prod=1500):
    if not (HAS_EMCEE and HAS_BATMAN):
        raise RuntimeError("Missing emcee and/or batman-package. Install them to run MCMC.")

    # Center time
    t_med = np.median(t)
    t_rel = t - t_med
    t_span = (np.min(t_rel), np.max(t_rel))

    P = fixed["P"]
    ecc = fixed["ecc"]
    w = fixed["w"]
    u1 = fixed["u1"]
    u2 = fixed["u2"]

    t0_guess_abs = guesses["t0"]
    t0_guess_rel = float(t0_guess_abs - t_med)

    theta0 = np.array([t0_guess_rel, guesses["rp"], guesses["a"], guesses["inc"], guesses["baseline"], math.log(max(guesses["jitter"], 1e-9))], float)

    def log_prior(theta):
        t0_rel, rp, a, inc, baseline, ln_jit = theta
        if not (t_span[0] <= t0_rel <= t_span[1]): return -np.inf
        if not (0.001 <= rp <= 0.3): return -np.inf
        if not (1.0 <= a <= 80.0): return -np.inf
        if not (60.0 <= inc <= 90.0): return -np.inf
        if not (0.5 <= baseline <= 1.5): return -np.inf
        if not (-20.0 <= ln_jit <= -1.0): return -np.inf
        return -0.5 * ((baseline - 1.0) / 0.05) ** 2

    def log_like(theta):
        t0_rel, rp, a, inc, baseline, ln_jit = theta
        model = batman_flux_model(t_rel, t0_rel, rp, a, inc, baseline, P, ecc, w, u1, u2)
        jitter = np.exp(ln_jit)
        s2 = yerr**2 + jitter**2
        return -0.5 * np.sum((y - model)**2 / s2 + np.log(2*np.pi*s2))

    def log_post(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_like(theta)

    rng = np.random.default_rng(0)
    scale = np.array([
        0.05 * (t_span[1] - t_span[0] + 1e-12),
        0.02 * max(guesses["rp"], 1e-3),
        1.0,
        0.5,
        0.01,
        0.5,
    ])
    p0 = theta0 + rng.normal(scale=scale, size=(walkers, theta0.size))

    sampler = emcee.EnsembleSampler(walkers, theta0.size, log_post)
    sampler.run_mcmc(p0, burn, progress=False)
    sampler.reset()
    sampler.run_mcmc(None, prod, progress=False)

    samples = sampler.get_chain(flat=True)
    samples_abs = samples.copy()
    samples_abs[:, 0] = samples[:, 0] + t_med

    acc = float(np.mean(sampler.acceptance_fraction))
    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau = np.asarray(tau, float)
    except Exception:
        tau = np.full(theta0.size, np.nan, dtype=float)
    tau_med = float(np.nanmedian(tau)) if np.any(np.isfinite(tau)) else np.nan
    n_eff = float(samples.shape[0] / max(2.0 * tau_med, 1.0)) if np.isfinite(tau_med) else np.nan

    return dict(
        samples=samples,
        samples_abs=samples_abs,
        t_med=t_med,
        fixed=fixed,
        acceptance_fraction=acc,
        autocorr_time=tau,
        n_eff=n_eff,
    )


# -------------------------
# -------------------------
# PySide6 + pyqtgraph App
# -------------------------

STAR_COLORS = {
    "target": "#E11D48",
    "comp1": "#0EA5E9",
    "comp2": "#10B981",
    "comp3": "#F59E0B",
    "comp4": "#8B5CF6",
}


def _gray_lut():
    lut = np.empty((256, 4), dtype=np.ubyte)
    vals = np.arange(256, dtype=np.ubyte)
    lut[:, 0] = vals
    lut[:, 1] = vals
    lut[:, 2] = vals
    lut[:, 3] = 255
    return lut


GRAY_LUT = _gray_lut()


class HoverImageWidget(QtWidgets.QWidget):
    hoverChanged = QtCore.Signal(float, float)

    def __init__(self, parent=None, title=""):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.plot = pg.PlotWidget()
        self.plot.getPlotItem().setTitle(title)
        self.plot.setBackground("w")
        self.plot.showGrid(x=False, y=False, alpha=0.2)
        self.plot.getPlotItem().setAspectLocked(True)
        self.plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)

        self.img_item = pg.ImageItem()
        self.img_item.setLookupTable(GRAY_LUT)
        self.img_item.setAutoDownsample(False)
        self.plot.addItem(self.img_item)

        self.scatter = pg.ScatterPlotItem()
        self.plot.addItem(self.scatter)

        self.cursor_text = pg.TextItem(anchor=(0, 1),color=(20, 30, 45),border=pg.mkPen("#64748B", width=1),fill=pg.mkBrush(255, 255, 255, 235),  # white box
)

        self.cursor_text.setZValue(20)
        self.plot.addItem(self.cursor_text)

        self.circle_items = []
        self._ny = None
        self._nx = None

        self._proxy = pg.SignalProxy(
            self.plot.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_mouse_moved,
        )

        lay.addWidget(self.plot)

    def clear(self):
        self.img_item.clear()
        self.scatter.clear()
        for it in self.circle_items:
            self.plot.removeItem(it)
        self.circle_items = []
        self._ny = None
        self._nx = None
        self.cursor_text.setText("")

    def _to_plot_y(self, y_data):
        if self._ny is None:
            return y_data
        return (self._ny - 1.0) - y_data

    def _to_data_y(self, y_plot):
        if self._ny is None:
            return y_plot
        return (self._ny - 1.0) - y_plot

    def set_image(self, img, vmin=None, vmax=None, auto_range=True):
        arr = np.asarray(img)
        if arr.ndim != 2:
            raise ValueError("Image must be 2D")
        self._ny, self._nx = arr.shape

        data = np.flipud(arr)

        if vmin is None or vmax is None:
            vmin_, vmax_ = percentile_vmin_vmax(arr)
        else:
            vmin_, vmax_ = float(vmin), float(vmax)

        self.img_item.setImage(data, autoLevels=False, levels=(vmin_, vmax_))
        self.img_item.setRect(QtCore.QRectF(0, 0, self._nx, self._ny))
        self.plot.setLabel("bottom", "x [pix]")
        self.plot.setLabel("left", "y [pix]")

        if auto_range:
            self.plot.setXRange(0, self._nx, padding=0.0)
            self.plot.setYRange(0, self._ny, padding=0.0)

    def set_points(self, points):
        spots = []
        for p in points:
            y_plot = self._to_plot_y(float(p["y"]))
            spots.append(
                {
                    "pos": (float(p["x"]), y_plot),
                    "size": float(p.get("size", 10)),
                    "symbol": p.get("symbol", "o"),
                    "pen": pg.mkPen(p.get("pen", "#0F172A"), width=2),
                    "brush": pg.mkBrush(p.get("brush", "#0F172A")),
                }
            )
        self.scatter.setData(spots)

    def set_circles(self, circles):
        for it in self.circle_items:
            self.plot.removeItem(it)
        self.circle_items = []

        th = np.linspace(0.0, 2.0 * np.pi, 256)
        for c in circles:
            cx = float(c["x"])
            cy = self._to_plot_y(float(c["y"]))
            r = float(c["r"])
            x = cx + r * np.cos(th)
            y = cy + r * np.sin(th)
            pen = pg.mkPen(c.get("color", "#22D3EE"), width=float(c.get("width", 1.6)), style=c.get("style", QtCore.Qt.SolidLine))
            item = pg.PlotDataItem(x, y, pen=pen)
            item.setZValue(9)
            self.plot.addItem(item)
            self.circle_items.append(item)

    def _on_mouse_moved(self, evt):
        pos = evt[0]
        if not self.plot.sceneBoundingRect().contains(pos):
            return

        vb = self.plot.getPlotItem().vb
        mouse_point = vb.mapSceneToView(pos)
        x = float(mouse_point.x())
        y_plot = float(mouse_point.y())
        y = float(self._to_data_y(y_plot))

        self.cursor_text.setText(f"x={x:.1f}, y={y:.1f}")
        self.cursor_text.setPos(x + 1.5, y_plot + 1.5)
        self.hoverChanged.emit(x, y)


class CornerDialog(QtWidgets.QDialog):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Corner plot")
        self.resize(820, 700)
        lay = QtWidgets.QVBoxLayout(self)
        canvas = FigureCanvas(fig)
        lay.addWidget(canvas)


class ExoTransitMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GRIFO — Graphical Reduction and Inference for exoplanetary transit Observations")
        self.setMinimumSize(980, 640)

        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            geom = screen.availableGeometry()
            win_w = min(max(1120, int(0.92 * geom.width())), geom.width())
            win_h = min(max(720, int(0.90 * geom.height())), geom.height())
            self.resize(win_w, win_h)
        else:
            self.resize(1400, 860)

        self.cube_raw = None
        self.cube_aligned = None
        self.images = []
        self.shifts_yx = None
        self.rotations_deg = None
        self.stars = self._default_stars()
        self.centroids_per_frame = None
        self.last_phot = None
        self.last_mcmc = None
        self.last_polyfit = None

        self._syncing_star_widgets = False
        self._manual_mcmc_guess = False

        self._build_ui()
        self._apply_style()
        self._update_all_for_no_data()

    # -------------------------
    # State helpers
    # -------------------------
    def _default_stars(self):
        return {
            "target": {"x": 100.0, "y": 100.0},
            "comp1": {"x": 120.0, "y": 120.0},
            "comp2": {"x": 140.0, "y": 140.0},
            "comp3": {"x": 160.0, "y": 160.0},
            "comp4": {"x": 180.0, "y": 180.0},
            "enabled": {"comp1": True, "comp2": False, "comp3": False, "comp4": False},
        }

    def active_cube(self):
        if self.cube_aligned is not None:
            return self.cube_aligned
        return self.cube_raw

    def active_cube_name(self):
        return "aligned" if self.cube_aligned is not None else "raw"

    def _require_data(self):
        if self.cube_raw is None or len(self.images) == 0:
            QtWidgets.QMessageBox.information(self, "No data", "Load FITS files first.")
            return False
        return True

    def _show_error(self, message):
        QtWidgets.QMessageBox.critical(self, "Error", str(message))

    def _show_info(self, message):
        self.statusBar().showMessage(str(message), 7000)

    def _set_busy(self, busy=True):
        self.setCursor(QtCore.Qt.WaitCursor if busy else QtCore.Qt.ArrowCursor)
        QtWidgets.QApplication.processEvents()

    # -------------------------
    # UI setup
    # -------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 10)
        root.setSpacing(10)

        # Header
        header_row = QtWidgets.QHBoxLayout()
        title_col = QtWidgets.QVBoxLayout()
        self.title_lbl = QtWidgets.QLabel("GRIFO — Graphical Reduction and Inference for exoplanetary transit Observations")
        self.title_lbl.setObjectName("titleLabel")
        self.title_lbl.setWordWrap(True)
        self.subtitle_lbl = QtWidgets.QLabel(
            "Desktop workflow for frame inspection, alignment, differential photometry, detrending, and model fitting."
        )
        self.subtitle_lbl.setObjectName("subtitleLabel")
        title_col.addWidget(self.title_lbl)
        title_col.addWidget(self.subtitle_lbl)

        credits_box = QtWidgets.QFrame()
        credits_box.setObjectName("creditsBox")
        credits_lay = QtWidgets.QVBoxLayout(credits_box)
        credits_lay.setContentsMargins(12, 10, 12, 10)
        credits_lay.setSpacing(2)
        credits_lay.addWidget(QtWidgets.QLabel("<b>Moreno Monticelli</b>"))
        credits_lay.addWidget(QtWidgets.QLabel("INAF-UniGe"))
        credits_lay.addWidget(QtWidgets.QLabel("moreno.monticelli@inaf.it"))

        header_row.addLayout(title_col, stretch=1)
        header_row.addWidget(credits_box, stretch=0, alignment=QtCore.Qt.AlignTop)
        root.addLayout(header_row)

        # Load and reset controls
        controls = QtWidgets.QFrame()
        controls.setObjectName("controlsCard")
        controls_lay = QtWidgets.QGridLayout(controls)
        controls_lay.setContentsMargins(12, 10, 12, 10)
        controls_lay.setHorizontalSpacing(14)
        controls_lay.setVerticalSpacing(8)

        self.load_btn = QtWidgets.QPushButton("Load FITS")
        self.load_btn.clicked.connect(self.load_fits_dialog)
        controls_lay.addWidget(self.load_btn, 0, 0)

        self.bin2x2_chk = QtWidgets.QCheckBox("Binning 2x2")
        self.bin2x2_chk.setChecked(True)
        controls_lay.addWidget(self.bin2x2_chk, 0, 1)

        controls_lay.addWidget(QtWidgets.QLabel("Data type"), 0, 2)
        self.dtype_combo = QtWidgets.QComboBox()
        self.dtype_combo.addItems(["float32", "float64"])
        controls_lay.addWidget(self.dtype_combo, 0, 3)

        self.reset_stars_btn = QtWidgets.QPushButton("Reset stars")
        self.reset_stars_btn.clicked.connect(self.reset_stars)
        controls_lay.addWidget(self.reset_stars_btn, 0, 4)

        self.reset_phot_btn = QtWidgets.QPushButton("Reset phot+detrend+fit")
        self.reset_phot_btn.clicked.connect(self.reset_photometry_products)
        controls_lay.addWidget(self.reset_phot_btn, 0, 5)

        self.reset_align_btn = QtWidgets.QPushButton("Reset alignment")
        self.reset_align_btn.clicked.connect(self.reset_alignment)
        controls_lay.addWidget(self.reset_align_btn, 0, 6)

        self.summary_text = QtWidgets.QLabel("No data loaded")
        self.summary_text.setObjectName("summaryText")
        controls_lay.addWidget(self.summary_text, 1, 0, 1, 7)

        self.mem_text = QtWidgets.QLabel("")
        self.mem_text.setObjectName("summaryText")
        controls_lay.addWidget(self.mem_text, 2, 0, 1, 7)

        root.addWidget(controls)

        # Workflow tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self.tabs, stretch=1)

        self._build_tab_inspect()
        self._build_tab_align()
        self._build_tab_stars()
        self._build_tab_photometry()
        self._build_tab_detrend()
        self._build_tab_mcmc()

        self.statusBar().showMessage("Ready")

    def _add_scroll_tab(self, content_widget, title):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setWidget(content_widget)
        self.tabs.addTab(scroll, title)

    def _apply_style(self):
        self.setStyleSheet(
            """
            QWidget {
                background: #F4F7FB;
                color: #0F172A;
                font-size: 12px;
            }
            QMainWindow::separator { background: #D8E0EA; }
            #titleLabel {
                font-size: 44px;
                font-weight: 800;
                color: #0B132F;
            }
            #subtitleLabel {
                font-size: 16px;
                color: #566579;
            }
            #creditsBox {
                background: rgba(255,255,255,0.94);
                border: 1px solid #D8E0EA;
                border-radius: 12px;
            }
            #controlsCard {
                background: #FFFFFF;
                border: 1px solid #D8E0EA;
                border-radius: 12px;
            }
            #summaryText {
                color: #465A73;
                font-size: 12px;
            }
            QGroupBox {
                font-weight: 700;
                border: 1px solid #D8E0EA;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 10px;
                background: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px 0 4px;
            }
            QTabWidget::pane {
                border: 1px solid #D8E0EA;
                border-radius: 12px;
                background: #FFFFFF;
                top: -1px;
            }
            QTabBar::tab {
                background: #EAF1F8;
                border: 1px solid #D8E0EA;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 8px 14px;
                margin-right: 3px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #FFFFFF;
                color: #0B6E99;
            }
            QPushButton {
                background: #FFFFFF;
                border: 1px solid #C7D5E5;
                border-radius: 8px;
                padding: 6px 10px;
                font-weight: 600;
            }
            QPushButton:hover { border-color: #7AA6C5; }
            QPushButton:pressed { background: #EAF2FA; }
            QProgressBar {
                border: 1px solid #C7D5E5;
                border-radius: 7px;
                text-align: center;
                background: #F7FAFE;
            }
            QProgressBar::chunk { background-color: #0B6E99; border-radius: 6px; }
            """
        )

    # -------------------------
    # Data load/reset
    # -------------------------
    def load_fits_dialog(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select FITS files",
            "",
            "FITS Files (*.fits *.fit *.fts)",
        )
        if not paths:
            return

        self._set_busy(True)
        try:
            cube, images = load_fits_cube_from_paths(
                paths,
                bin2x2=self.bin2x2_chk.isChecked(),
                dtype=self.dtype_combo.currentText(),
            )
        except Exception as exc:
            self._show_error(exc)
            self._set_busy(False)
            return

        self.cube_raw = cube
        self.cube_aligned = None
        self.images = images
        self.shifts_yx = None
        self.rotations_deg = None
        self.centroids_per_frame = None
        self.last_phot = None
        self.last_mcmc = None
        self.last_polyfit = None

        self._set_busy(False)
        self._show_info(f"Loaded {cube.shape[0]} frames. Cube shape: {cube.shape}")
        self._refresh_after_data_change()

    def reset_stars(self):
        self.stars = self._default_stars()
        self.centroids_per_frame = None
        self.sync_star_widgets_from_state()
        self.update_stars_view()
        self.update_phot_inspect_options()
        self.update_phot_cutout()
        self._show_info("Star selections reset.")

    def reset_photometry_products(self):
        self.last_phot = None
        self.last_mcmc = None
        self.last_polyfit = None
        self.update_phot_plots()
        self.update_detrend_plots()
        self.update_mcmc_preview_plot()
        self.update_mcmc_result_plot()
        self._show_info("Photometry, detrend, and fit products reset.")

    def reset_alignment(self):
        self.cube_aligned = None
        self.shifts_yx = None
        self.rotations_deg = None
        self.update_alignment_plots()
        self._refresh_after_data_change()
        self._show_info("Alignment reset.")

    def _refresh_after_data_change(self):
        cube = self.active_cube()
        if cube is None:
            self._update_all_for_no_data()
            return

        n = int(cube.shape[0])
        self.align_ref_idx.setRange(0, max(0, n - 1))
        self.align_max_frames.setRange(0, n)
        self.align_max_frames.setValue(n)

        self.stars_frame_slider.setRange(0, max(0, n - 1))
        self.stars_frame_spin.setRange(0, max(0, n - 1))
        self.stars_max_frames.setRange(0, n)
        self.stars_max_frames.setValue(n)

        self.phot_preview_slider.setRange(0, max(0, n - 1))
        self.phot_preview_spin.setRange(0, max(0, n - 1))
        self.phot_max_frames.setRange(0, n)
        self.phot_max_frames.setValue(n)

        ny, nx = cube.shape[1], cube.shape[2]
        for name in ["target", "comp1", "comp2", "comp3", "comp4"]:
            self.star_x_spin[name].setRange(0.0, float(nx - 1))
            self.star_y_spin[name].setRange(0.0, float(ny - 1))

        self.update_summary_block()
        self.update_inspect_filter_options()
        self.update_inspect_view()
        self.update_alignment_plots()
        self.sync_star_widgets_from_state()
        self.update_stars_view()
        self.update_phot_inspect_options()
        self.update_phot_cutout()
        self.update_phot_plots()
        self.update_detrend_defaults()
        self.update_detrend_plots()
        self.update_mcmc_guess_defaults(force=True)
        self.update_mcmc_preview_plot()
        self.update_mcmc_result_plot()

    def _update_all_for_no_data(self):
        self.summary_text.setText("No data loaded")
        self.mem_text.setText("")
        self.inspect_image.clear()
        self.stars_image.clear()
        self.phot_cutout_image.clear()
        self.align_dx_plot.clear()
        self.align_dy_plot.clear()
        self.align_rot_plot.clear()
        self.phot_rel_plot.clear()
        self.phot_raw_plot.clear()
        self.det_plot.clear()
        self.det_trend_plot.clear()
        self.mcmc_preview_plot.clear()
        self.mcmc_result_plot.clear()
        self.mcmc_summary_box.setPlainText("No fit result yet.")

    def update_summary_block(self):
        cube = self.active_cube()
        if cube is None:
            self.summary_text.setText("No data loaded")
            self.mem_text.setText("")
            return

        diag = time_axis_diagnostics(self.images)
        n_filters = len({im.get("filter", "NA") for im in self.images})
        summary = (
            f"Frames: {cube.shape[0]}  |  Filters: {n_filters}  |  "
            f"Missing JD: {diag['n_missing_jd']}  |  Duplicate JD: {diag['n_duplicate_jd']}  |  "
            f"BJD-like headers: {diag['n_bjd_like']}"
        )
        self.summary_text.setText(summary)

        raw_mb = float(self.cube_raw.nbytes) / (1024.0 ** 2) if self.cube_raw is not None else 0.0
        aligned_mb = float(self.cube_aligned.nbytes) / (1024.0 ** 2) if self.cube_aligned is not None else 0.0
        total_mb = raw_mb + aligned_mb
        self.mem_text.setText(
            f"Approx cube RAM: raw={raw_mb:.1f} MB, aligned={aligned_mb:.1f} MB, total={total_mb:.1f} MB"
        )

        if diag["n_non_monotonic"] > 0:
            self.statusBar().showMessage(
                f"Time axis not monotonic in {diag['n_non_monotonic']} pairs; data are sorted before photometry.",
                9000,
            )

    # -------------------------
    # Tab: Inspect
    # -------------------------
    def _build_tab_inspect(self):
        tab = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(tab)

        left = QtWidgets.QGroupBox("Inspect controls")
        left_lay = QtWidgets.QFormLayout(left)

        self.inspect_filter = QtWidgets.QComboBox()
        self.inspect_filter.currentTextChanged.connect(self.update_inspect_frame_range)
        left_lay.addRow("Filter", self.inspect_filter)

        self.inspect_frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.inspect_frame_slider.valueChanged.connect(self._sync_inspect_spin)
        self.inspect_frame_slider.valueChanged.connect(self.update_inspect_view)
        left_lay.addRow("Frame idx", self.inspect_frame_slider)

        self.inspect_frame_spin = QtWidgets.QSpinBox()
        self.inspect_frame_spin.valueChanged.connect(self.inspect_frame_slider.setValue)
        left_lay.addRow("Frame idx (spin)", self.inspect_frame_spin)

        self.inspect_log = QtWidgets.QCheckBox("Log scale")
        self.inspect_log.stateChanged.connect(self.update_inspect_view)
        left_lay.addRow(self.inspect_log)

        self.inspect_pmin = QtWidgets.QDoubleSpinBox()
        self.inspect_pmin.setRange(0.0, 20.0)
        self.inspect_pmin.setSingleStep(0.5)
        self.inspect_pmin.setValue(5.0)
        self.inspect_pmin.valueChanged.connect(self.update_inspect_view)
        left_lay.addRow("vmin percentile", self.inspect_pmin)

        self.inspect_pmax = QtWidgets.QDoubleSpinBox()
        self.inspect_pmax.setRange(80.0, 100.0)
        self.inspect_pmax.setSingleStep(0.1)
        self.inspect_pmax.setValue(99.5)
        self.inspect_pmax.valueChanged.connect(self.update_inspect_view)
        left_lay.addRow("vmax percentile", self.inspect_pmax)

        self.inspect_file = QtWidgets.QLabel("-")
        self.inspect_filter_lbl = QtWidgets.QLabel("-")
        self.inspect_jd = QtWidgets.QLabel("-")
        self.inspect_airmass = QtWidgets.QLabel("-")
        self.inspect_exptime = QtWidgets.QLabel("-")
        left_lay.addRow("File", self.inspect_file)
        left_lay.addRow("Filter", self.inspect_filter_lbl)
        left_lay.addRow("JD", self.inspect_jd)
        left_lay.addRow("Airmass", self.inspect_airmass)
        left_lay.addRow("Exposure [s]", self.inspect_exptime)

        right = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right)
        self.inspect_image = HoverImageWidget(title="Inspect frame")
        right_lay.addWidget(self.inspect_image)
        tip = QtWidgets.QLabel("Tip: verify target/comparison stars are visible and not saturated.")
        tip.setObjectName("summaryText")
        right_lay.addWidget(tip)

        root.addWidget(left, stretch=0)
        root.addWidget(right, stretch=1)
        self._add_scroll_tab(tab, "Inspect frames")

    def _sync_inspect_spin(self, value):
        if self.inspect_frame_spin.value() != value:
            self.inspect_frame_spin.blockSignals(True)
            self.inspect_frame_spin.setValue(value)
            self.inspect_frame_spin.blockSignals(False)

    def update_inspect_filter_options(self):
        if self.active_cube() is None:
            self.inspect_filter.blockSignals(True)
            self.inspect_filter.clear()
            self.inspect_filter.blockSignals(False)
            return

        filt = ["All"] + sorted({im.get("filter", "NA") for im in self.images})
        current = self.inspect_filter.currentText()
        self.inspect_filter.blockSignals(True)
        self.inspect_filter.clear()
        self.inspect_filter.addItems(filt)
        if current in filt:
            self.inspect_filter.setCurrentText(current)
        self.inspect_filter.blockSignals(False)
        self.update_inspect_frame_range()

    def _inspect_indices(self):
        cube = self.active_cube()
        if cube is None:
            return np.array([], dtype=int)
        selected = self.inspect_filter.currentText() or "All"
        if selected == "All":
            return np.arange(cube.shape[0], dtype=int)
        return np.array([i for i, im in enumerate(self.images) if im.get("filter", "NA") == selected], dtype=int)

    def update_inspect_frame_range(self):
        idx_list = self._inspect_indices()
        if idx_list.size == 0:
            self.inspect_frame_slider.setRange(0, 0)
            self.inspect_frame_spin.setRange(0, 0)
            return
        max_idx = int(idx_list.size - 1)
        self.inspect_frame_slider.setRange(0, max_idx)
        self.inspect_frame_spin.setRange(0, max_idx)
        self.update_inspect_view()

    def update_inspect_view(self):
        cube = self.active_cube()
        if cube is None:
            self.inspect_image.clear()
            return

        idx_list = self._inspect_indices()
        if idx_list.size == 0:
            self.inspect_image.clear()
            return

        idx_in_list = int(self.inspect_frame_slider.value())
        idx_in_list = np.clip(idx_in_list, 0, idx_list.size - 1)
        idx = int(idx_list[idx_in_list])

        img = cube[idx]
        vmin, vmax = percentile_vmin_vmax(
            img,
            pmin=float(self.inspect_pmin.value()),
            pmax=float(self.inspect_pmax.value()),
        )
        disp = img
        if self.inspect_log.isChecked():
            disp = np.clip(img, max(vmin, 1e-6), None)
            disp = np.log10(disp)
            vmin, vmax = np.nanpercentile(disp, 5), np.nanpercentile(disp, 99.5)

        self.inspect_image.set_image(disp, vmin=vmin, vmax=vmax)
        self.inspect_image.plot.getPlotItem().setTitle(f"Frame {idx} ({self.active_cube_name()} cube)")

        im = self.images[idx]
        self.inspect_file.setText(os.path.basename(str(im.get("file", ""))))
        self.inspect_filter_lbl.setText(str(im.get("filter", "NA")))
        self.inspect_jd.setText(f"{im.get('jd', np.nan):.8f}" if np.isfinite(_safe_float(im.get("jd", np.nan))) else "NaN")
        self.inspect_airmass.setText(f"{im.get('airmass', np.nan):.4f}" if np.isfinite(_safe_float(im.get("airmass", np.nan))) else "NaN")
        self.inspect_exptime.setText(f"{im.get('exptime', np.nan):.3f}" if np.isfinite(_safe_float(im.get("exptime", np.nan))) else "NaN")

    # -------------------------
    # Tab: Align
    # -------------------------
    def _build_tab_align(self):
        tab = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(tab)

        left = QtWidgets.QGroupBox("Alignment settings")
        form = QtWidgets.QFormLayout(left)

        self.align_ref_idx = QtWidgets.QSpinBox()
        form.addRow("Reference frame", self.align_ref_idx)

        self.align_fwhm = QtWidgets.QDoubleSpinBox()
        self.align_fwhm.setRange(0.5, 30.0)
        self.align_fwhm.setValue(3.0)
        self.align_fwhm.setSingleStep(0.5)
        form.addRow("DAOStarFinder FWHM", self.align_fwhm)

        self.align_thr = QtWidgets.QDoubleSpinBox()
        self.align_thr.setRange(0.5, 50.0)
        self.align_thr.setValue(5.0)
        self.align_thr.setSingleStep(0.5)
        form.addRow("Threshold sigma", self.align_thr)

        self.align_brightest = QtWidgets.QSpinBox()
        self.align_brightest.setRange(10, 5000)
        self.align_brightest.setValue(150)
        self.align_brightest.setSingleStep(10)
        form.addRow("Keep brightest stars", self.align_brightest)

        self.align_binsize = QtWidgets.QDoubleSpinBox()
        self.align_binsize.setRange(0.5, 20.0)
        self.align_binsize.setValue(2.0)
        self.align_binsize.setSingleStep(0.5)
        form.addRow("Histogram binsize [px]", self.align_binsize)

        self.align_max_shift = QtWidgets.QDoubleSpinBox()
        self.align_max_shift.setRange(5.0, 5000.0)
        self.align_max_shift.setValue(200.0)
        self.align_max_shift.setSingleStep(10.0)
        form.addRow("Max shift [px]", self.align_max_shift)

        self.align_model = QtWidgets.QComboBox()
        self.align_model.addItems(["translation", "translation+rotation"])
        form.addRow("Alignment model", self.align_model)

        self.align_max_rot = QtWidgets.QDoubleSpinBox()
        self.align_max_rot.setRange(0.0, 90.0)
        self.align_max_rot.setValue(5.0)
        self.align_max_rot.setSingleStep(0.5)
        form.addRow("Max rotation [deg]", self.align_max_rot)

        self.align_order = QtWidgets.QComboBox()
        self.align_order.addItems(["0", "1", "3"])
        self.align_order.setCurrentText("3")
        form.addRow("Interpolation order", self.align_order)

        self.align_max_frames = QtWidgets.QSpinBox()
        form.addRow("Max frames to process", self.align_max_frames)

        self.align_replace_raw = QtWidgets.QCheckBox("Replace raw cube with aligned cube (save RAM)")
        form.addRow(self.align_replace_raw)

        self.align_run_btn = QtWidgets.QPushButton("Run alignment")
        self.align_run_btn.clicked.connect(self.run_alignment)
        form.addRow(self.align_run_btn)

        self.align_prog = QtWidgets.QProgressBar()
        self.align_prog.setRange(0, 100)
        form.addRow("Progress", self.align_prog)

        self.align_status = QtWidgets.QLabel("Idle")
        self.align_status.setObjectName("summaryText")
        form.addRow(self.align_status)

        right = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right)
        self.align_dx_plot = pg.PlotWidget(title="Estimated dx shifts")
        self.align_dy_plot = pg.PlotWidget(title="Estimated dy shifts")
        self.align_rot_plot = pg.PlotWidget(title="Estimated fallback rotations")
        for pw in [self.align_dx_plot, self.align_dy_plot, self.align_rot_plot]:
            pw.setBackground("w")
            pw.showGrid(x=True, y=True, alpha=0.25)
            right_lay.addWidget(pw)

        root.addWidget(left, stretch=0)
        root.addWidget(right, stretch=1)
        self._add_scroll_tab(tab, "Align frames")

    def run_alignment(self):
        if not self._require_data():
            return

        cube_raw = self.cube_raw
        n = int(cube_raw.shape[0])
        max_frames = int(self.align_max_frames.value())
        max_frames = max_frames if max_frames > 0 else None

        self._set_busy(True)
        t0_run = time.time()
        self.align_prog.setValue(0)

        def cb(i, n_tot, dy, dx, rot):
            self.align_prog.setValue(int(100 * i / max(n_tot, 1)))
            elapsed = time.time() - t0_run
            eta = (elapsed / max(i, 1)) * max(n_tot - i, 0)
            self.align_status.setText(
                f"Aligning {i}/{n_tot} | dy={dy:.2f} dx={dx:.2f} rot={rot:.2f} deg | elapsed={elapsed:.1f}s eta={eta:.1f}s"
            )
            QtWidgets.QApplication.processEvents()

        try:
            aligned, shifts, rots = align_cube_translation(
                cube_raw,
                ref_index=int(self.align_ref_idx.value()),
                fwhm=float(self.align_fwhm.value()),
                thr_sigma=float(self.align_thr.value()),
                brightest=int(self.align_brightest.value()),
                binsize=float(self.align_binsize.value()),
                max_shift=float(self.align_max_shift.value()),
                order=int(self.align_order.currentText()),
                progress_cb=cb,
                model=str(self.align_model.currentText()),
                max_rotation_deg=float(self.align_max_rot.value()),
                max_frames=max_frames,
            )
        except Exception as exc:
            self._set_busy(False)
            self._show_error(exc)
            return

        if self.align_replace_raw.isChecked():
            self.cube_raw = aligned
            self.cube_aligned = None
            self._show_info("Alignment complete. Raw cube replaced to save RAM.")
        else:
            self.cube_aligned = aligned
            self._show_info("Alignment complete. The workflow now uses the aligned cube.")

        self.shifts_yx = shifts
        self.rotations_deg = rots

        self.align_status.setText(f"Alignment done ({n} frames).")
        self.align_prog.setValue(100)
        self._set_busy(False)
        self._refresh_after_data_change()

    def update_alignment_plots(self):
        for pw in [self.align_dx_plot, self.align_dy_plot, self.align_rot_plot]:
            pw.clear()

        if self.shifts_yx is None:
            self.align_status.setText("Run alignment to see diagnostics.")
            return

        x = np.arange(self.shifts_yx.shape[0], dtype=float)
        dx = np.asarray(self.shifts_yx[:, 1], float)
        dy = np.asarray(self.shifts_yx[:, 0], float)
        rot = np.asarray(self.rotations_deg, float) if self.rotations_deg is not None else np.zeros_like(dx)

        self.align_dx_plot.plot(x, dx, pen=pg.mkPen("#0EA5E9", width=2), symbol="o", symbolSize=5, symbolBrush="#0EA5E9")
        self.align_dx_plot.setLabel("left", "dx [px]")
        self.align_dx_plot.setLabel("bottom", "Frame")

        self.align_dy_plot.plot(x, dy, pen=pg.mkPen("#10B981", width=2), symbol="o", symbolSize=5, symbolBrush="#10B981")
        self.align_dy_plot.setLabel("left", "dy [px]")
        self.align_dy_plot.setLabel("bottom", "Frame")

        self.align_rot_plot.plot(x, rot, pen=pg.mkPen("#F59E0B", width=2), symbol="o", symbolSize=5, symbolBrush="#F59E0B")
        self.align_rot_plot.setLabel("left", "Rotation [deg]")
        self.align_rot_plot.setLabel("bottom", "Frame")

    # -------------------------
    # Tab: Stars
    # -------------------------
    def _build_tab_stars(self):
        tab = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(tab)

        left = QtWidgets.QGroupBox("Star coordinates (manual)")
        left_lay = QtWidgets.QVBoxLayout(left)

        frm = QtWidgets.QFormLayout()

        self.stars_frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.stars_frame_slider.valueChanged.connect(self._sync_stars_frame_spin)
        self.stars_frame_slider.valueChanged.connect(self.update_stars_view)
        frm.addRow("Frame for visual check", self.stars_frame_slider)

        self.stars_frame_spin = QtWidgets.QSpinBox()
        self.stars_frame_spin.valueChanged.connect(self.stars_frame_slider.setValue)
        frm.addRow("Frame index", self.stars_frame_spin)

        self.stars_half = QtWidgets.QSpinBox()
        self.stars_half.setRange(3, 30)
        self.stars_half.setValue(10)
        frm.addRow("Centroid refine half-size", self.stars_half)

        self.stars_coord_step = QtWidgets.QComboBox()
        self.stars_coord_step.addItems(["1.0", "0.5", "0.1"])
        self.stars_coord_step.setCurrentText("0.1")
        self.stars_coord_step.currentTextChanged.connect(self._apply_star_coord_step)
        frm.addRow("Coordinate input step [px]", self.stars_coord_step)

        self.stars_log = QtWidgets.QCheckBox("Log scale")
        self.stars_log.stateChanged.connect(self.update_stars_view)
        frm.addRow(self.stars_log)

        self.stars_pmin = QtWidgets.QDoubleSpinBox()
        self.stars_pmin.setRange(0.0, 20.0)
        self.stars_pmin.setValue(5.0)
        self.stars_pmin.setSingleStep(0.5)
        self.stars_pmin.valueChanged.connect(self.update_stars_view)
        frm.addRow("vmin percentile", self.stars_pmin)

        self.stars_pmax = QtWidgets.QDoubleSpinBox()
        self.stars_pmax.setRange(80.0, 100.0)
        self.stars_pmax.setValue(99.5)
        self.stars_pmax.setSingleStep(0.1)
        self.stars_pmax.valueChanged.connect(self.update_stars_view)
        frm.addRow("vmax percentile", self.stars_pmax)

        self.stars_max_frames = QtWidgets.QSpinBox()
        frm.addRow("Max frames for all-frame refine", self.stars_max_frames)

        left_lay.addLayout(frm)

        target_box = QtWidgets.QGroupBox("Target")
        tf = QtWidgets.QFormLayout(target_box)

        self.star_x_spin = {}
        self.star_y_spin = {}
        self.star_enable = {}

        self.star_x_spin["target"] = QtWidgets.QDoubleSpinBox()
        self.star_y_spin["target"] = QtWidgets.QDoubleSpinBox()
        tf.addRow("Target x", self.star_x_spin["target"])
        tf.addRow("Target y", self.star_y_spin["target"])
        left_lay.addWidget(target_box)

        comps_box = QtWidgets.QGroupBox("Comparison stars")
        cf = QtWidgets.QFormLayout(comps_box)
        for name in ["comp1", "comp2", "comp3", "comp4"]:
            row = QtWidgets.QWidget()
            row_lay = QtWidgets.QHBoxLayout(row)
            row_lay.setContentsMargins(0, 0, 0, 0)
            en = QtWidgets.QCheckBox(name)
            self.star_enable[name] = en
            xsp = QtWidgets.QDoubleSpinBox()
            ysp = QtWidgets.QDoubleSpinBox()
            self.star_x_spin[name] = xsp
            self.star_y_spin[name] = ysp
            row_lay.addWidget(en)
            row_lay.addWidget(QtWidgets.QLabel("x"))
            row_lay.addWidget(xsp)
            row_lay.addWidget(QtWidgets.QLabel("y"))
            row_lay.addWidget(ysp)
            cf.addRow(row)
        left_lay.addWidget(comps_box)

        for name in ["target", "comp1", "comp2", "comp3", "comp4"]:
            self.star_x_spin[name].setDecimals(3)
            self.star_y_spin[name].setDecimals(3)
            self.star_x_spin[name].valueChanged.connect(self.on_star_widgets_changed)
            self.star_y_spin[name].valueChanged.connect(self.on_star_widgets_changed)
        for name in ["comp1", "comp2", "comp3", "comp4"]:
            self.star_enable[name].stateChanged.connect(self.on_star_widgets_changed)

        btn_row = QtWidgets.QHBoxLayout()
        self.stars_round_btn = QtWidgets.QPushButton("Round all coordinates")
        self.stars_round_btn.clicked.connect(self.round_star_coords)
        self.stars_refine_this_btn = QtWidgets.QPushButton("Refine on THIS frame")
        self.stars_refine_this_btn.clicked.connect(self.refine_stars_this_frame)
        self.stars_refine_all_btn = QtWidgets.QPushButton("Refine on ALL frames")
        self.stars_refine_all_btn.clicked.connect(self.refine_stars_all_frames)
        btn_row.addWidget(self.stars_round_btn)
        btn_row.addWidget(self.stars_refine_this_btn)
        btn_row.addWidget(self.stars_refine_all_btn)
        left_lay.addLayout(btn_row)

        self.stars_prog = QtWidgets.QProgressBar()
        self.stars_prog.setRange(0, 100)
        self.stars_status = QtWidgets.QLabel("Manual coordinate mode. Hover image for x,y.")
        self.stars_status.setObjectName("summaryText")
        left_lay.addWidget(self.stars_prog)
        left_lay.addWidget(self.stars_status)

        right = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right)
        self.stars_image = HoverImageWidget(title="Frame with chosen stars")
        self.stars_image.hoverChanged.connect(self._stars_hover_text)
        right_lay.addWidget(self.stars_image)
        self.stars_hover_lbl = QtWidgets.QLabel("Hover anywhere to read x,y. Enter coordinates manually in the left panel.")
        self.stars_hover_lbl.setObjectName("summaryText")
        right_lay.addWidget(self.stars_hover_lbl)

        root.addWidget(left, stretch=0)
        root.addWidget(right, stretch=1)
        self._add_scroll_tab(tab, "Pick stars")

        self._apply_star_coord_step()

    def _sync_stars_frame_spin(self, value):
        if self.stars_frame_spin.value() != value:
            self.stars_frame_spin.blockSignals(True)
            self.stars_frame_spin.setValue(value)
            self.stars_frame_spin.blockSignals(False)

    def _apply_star_coord_step(self):
        step = float(self.stars_coord_step.currentText())
        dec = 1 if step >= 1 else 3
        for name in ["target", "comp1", "comp2", "comp3", "comp4"]:
            self.star_x_spin[name].setSingleStep(step)
            self.star_y_spin[name].setSingleStep(step)
            self.star_x_spin[name].setDecimals(dec)
            self.star_y_spin[name].setDecimals(dec)

    def sync_star_widgets_from_state(self):
        self._syncing_star_widgets = True
        try:
            for name in ["target", "comp1", "comp2", "comp3", "comp4"]:
                self.star_x_spin[name].setValue(float(self.stars[name]["x"]))
                self.star_y_spin[name].setValue(float(self.stars[name]["y"]))
            for name in ["comp1", "comp2", "comp3", "comp4"]:
                self.star_enable[name].setChecked(bool(self.stars["enabled"][name]))
        finally:
            self._syncing_star_widgets = False

    def on_star_widgets_changed(self):
        if self._syncing_star_widgets:
            return

        for name in ["target", "comp1", "comp2", "comp3", "comp4"]:
            self.stars[name]["x"] = float(self.star_x_spin[name].value())
            self.stars[name]["y"] = float(self.star_y_spin[name].value())
        for name in ["comp1", "comp2", "comp3", "comp4"]:
            self.stars["enabled"][name] = bool(self.star_enable[name].isChecked())

        self.update_stars_view()
        self.update_phot_inspect_options()
        self.update_phot_cutout()

    def round_star_coords(self):
        for name in ["target", "comp1", "comp2", "comp3", "comp4"]:
            self.stars[name]["x"] = float(np.round(self.stars[name]["x"]))
            self.stars[name]["y"] = float(np.round(self.stars[name]["y"]))
        self.sync_star_widgets_from_state()
        self.update_stars_view()
        self._show_info("Rounded all coordinates to integer pixels.")

    def _stars_hover_text(self, x, y):
        self.stars_hover_lbl.setText(f"Hover x={x:.1f}, y={y:.1f}  |  Enter coordinates manually in the left panel.")

    def _star_overlay_points(self):
        points = [
            {
                "name": "target",
                "x": self.stars["target"]["x"],
                "y": self.stars["target"]["y"],
                "symbol": "x",
                "size": 16,
                "pen": STAR_COLORS["target"],
                "brush": STAR_COLORS["target"],
            }
        ]
        symbols = {"comp1": "o", "comp2": "d", "comp3": "t", "comp4": "s"}
        for name in ["comp1", "comp2", "comp3", "comp4"]:
            if self.stars["enabled"][name]:
                points.append(
                    {
                        "name": name,
                        "x": self.stars[name]["x"],
                        "y": self.stars[name]["y"],
                        "symbol": symbols[name],
                        "size": 13,
                        "pen": STAR_COLORS[name],
                        "brush": STAR_COLORS[name],
                    }
                )
        return points

    def update_stars_view(self):
        cube = self.active_cube()
        if cube is None:
            self.stars_image.clear()
            return

        idx = int(self.stars_frame_slider.value())
        idx = int(np.clip(idx, 0, cube.shape[0] - 1))
        img = cube[idx]
        vmin, vmax = percentile_vmin_vmax(
            img,
            pmin=float(self.stars_pmin.value()),
            pmax=float(self.stars_pmax.value()),
        )
        disp = img
        if self.stars_log.isChecked():
            disp = np.clip(img, max(vmin, 1e-6), None)
            disp = np.log10(disp)
            vmin, vmax = np.nanpercentile(disp, 5), np.nanpercentile(disp, 99.5)

        self.stars_image.set_image(disp, vmin=vmin, vmax=vmax)
        self.stars_image.set_points(self._star_overlay_points())
        self.stars_image.plot.getPlotItem().setTitle(f"Frame {idx} with chosen stars ({self.active_cube_name()} cube)")

    def refine_stars_this_frame(self):
        if not self._require_data():
            return

        cube = self.active_cube()
        idx = int(np.clip(self.stars_frame_slider.value(), 0, cube.shape[0] - 1))
        half = int(self.stars_half.value())
        img = cube[idx]

        xt, yt = centroid_2d(img, self.stars["target"]["x"], self.stars["target"]["y"], half_size=half)
        self.stars["target"]["x"] = float(xt)
        self.stars["target"]["y"] = float(yt)

        for name in ["comp1", "comp2", "comp3", "comp4"]:
            if self.stars["enabled"][name]:
                xc, yc = centroid_2d(img, self.stars[name]["x"], self.stars[name]["y"], half_size=half)
                self.stars[name]["x"] = float(xc)
                self.stars[name]["y"] = float(yc)

        self.sync_star_widgets_from_state()
        self.update_stars_view()
        self.update_phot_cutout()
        self._show_info("Refined centroids on current frame.")

    def refine_stars_all_frames(self):
        if not self._require_data():
            return

        cube = self.active_cube()
        n_total = int(cube.shape[0])
        n_loop = int(self.stars_max_frames.value()) if int(self.stars_max_frames.value()) > 0 else n_total
        n_loop = min(n_loop, n_total)
        half = int(self.stars_half.value())

        star_names = ["target", "comp1", "comp2", "comp3", "comp4"]
        per_frame = {name: np.full((n_total, 2), np.nan, dtype=np.float32) for name in star_names}

        t0_loop = time.time()
        self.stars_prog.setValue(0)
        self._set_busy(True)
        for i in range(n_loop):
            img = cube[i]
            per_frame["target"][i] = centroid_2d(img, self.stars["target"]["x"], self.stars["target"]["y"], half_size=half)
            for name in ["comp1", "comp2", "comp3", "comp4"]:
                if self.stars["enabled"][name]:
                    per_frame[name][i] = centroid_2d(img, self.stars[name]["x"], self.stars[name]["y"], half_size=half)

            self.stars_prog.setValue(int(100 * (i + 1) / max(n_loop, 1)))
            elapsed = time.time() - t0_loop
            eta = (elapsed / max(i + 1, 1)) * max(n_loop - (i + 1), 0)
            self.stars_status.setText(f"Processed {i + 1}/{n_loop} | elapsed={elapsed:.1f}s eta={eta:.1f}s")
            QtWidgets.QApplication.processEvents()

        self.centroids_per_frame = per_frame
        self._set_busy(False)
        self._show_info("Stored per-frame refined centroids.")

    # -------------------------
    # Tab: Photometry
    # -------------------------
    def _build_tab_photometry(self):
        tab = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(tab)

        # Left (controls)
        left = QtWidgets.QGroupBox("Photometry controls")
        left_outer = QtWidgets.QVBoxLayout(left)

        form = QtWidgets.QFormLayout()

        self.phot_preview_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.phot_preview_slider.valueChanged.connect(self._sync_phot_preview_spin)
        self.phot_preview_slider.valueChanged.connect(self.update_phot_cutout)
        form.addRow("Preview frame", self.phot_preview_slider)

        self.phot_preview_spin = QtWidgets.QSpinBox()
        self.phot_preview_spin.valueChanged.connect(self.phot_preview_slider.setValue)
        form.addRow("Preview frame (spin)", self.phot_preview_spin)

        self.phot_inspect_star = QtWidgets.QComboBox()
        self.phot_inspect_star.currentTextChanged.connect(self.update_phot_cutout)
        form.addRow("Cutout around", self.phot_inspect_star)

        self.phot_r_ap = QtWidgets.QDoubleSpinBox(); self.phot_r_ap.setRange(0.1, 200.0); self.phot_r_ap.setValue(8.0)
        self.phot_r_in = QtWidgets.QDoubleSpinBox(); self.phot_r_in.setRange(0.2, 300.0); self.phot_r_in.setValue(12.0)
        self.phot_r_out = QtWidgets.QDoubleSpinBox(); self.phot_r_out.setRange(0.3, 400.0); self.phot_r_out.setValue(18.0)
        form.addRow("Aperture radius r_ap [px]", self.phot_r_ap)
        form.addRow("Annulus inner r_in [px]", self.phot_r_in)
        form.addRow("Annulus outer r_out [px]", self.phot_r_out)

        self.phot_bkg_stat = QtWidgets.QComboBox(); self.phot_bkg_stat.addItems(["median", "mean"])
        self.phot_bkg_model = QtWidgets.QComboBox(); self.phot_bkg_model.addItems(["annulus", "plane"])
        self.phot_subpixels = QtWidgets.QComboBox(); self.phot_subpixels.addItems(["1", "3", "5", "7"]); self.phot_subpixels.setCurrentText("5")
        self.phot_comp_mode = QtWidgets.QComboBox(); self.phot_comp_mode.addItems(["weighted", "median", "mean", "sum"])
        form.addRow("Background estimator", self.phot_bkg_stat)
        form.addRow("Background model", self.phot_bkg_model)
        form.addRow("Subpixel sampling", self.phot_subpixels)
        form.addRow("Combine comparison stars", self.phot_comp_mode)

        self.phot_comp_sigma = QtWidgets.QDoubleSpinBox(); self.phot_comp_sigma.setRange(0.0, 20.0); self.phot_comp_sigma.setValue(3.0); self.phot_comp_sigma.setSingleStep(0.5)
        self.phot_noise_floor = QtWidgets.QDoubleSpinBox(); self.phot_noise_floor.setRange(0.0, 1e6); self.phot_noise_floor.setValue(0.0); self.phot_noise_floor.setSingleStep(50.0)
        form.addRow("Comparison clipping sigma (0=off)", self.phot_comp_sigma)
        form.addRow("Systematic noise floor [ppm]", self.phot_noise_floor)

        self.phot_edge_margin = QtWidgets.QDoubleSpinBox(); self.phot_edge_margin.setRange(0.0, 500.0); self.phot_edge_margin.setValue(5.0)
        self.phot_saturation = QtWidgets.QDoubleSpinBox(); self.phot_saturation.setRange(0.0, 1e9); self.phot_saturation.setValue(0.0); self.phot_saturation.setSingleStep(1000.0)
        self.phot_drop_bad = QtWidgets.QCheckBox("Drop frames failing quality checks"); self.phot_drop_bad.setChecked(True)
        self.phot_max_frames = QtWidgets.QSpinBox()
        self.phot_discard_nan_t = QtWidgets.QCheckBox("Drop frames with NaN JD"); self.phot_discard_nan_t.setChecked(True)

        form.addRow("Minimum edge margin [px]", self.phot_edge_margin)
        form.addRow("Saturation threshold [ADU] (0=off)", self.phot_saturation)
        form.addRow(self.phot_drop_bad)
        form.addRow("Max frames for photometry", self.phot_max_frames)
        form.addRow(self.phot_discard_nan_t)

        self.phot_cut_half = QtWidgets.QSpinBox(); self.phot_cut_half.setRange(10, 150); self.phot_cut_half.setValue(50)
        self.phot_cut_half.valueChanged.connect(self.update_phot_cutout)
        self.phot_log_cut = QtWidgets.QCheckBox("Log stretch (cutout)"); self.phot_log_cut.stateChanged.connect(self.update_phot_cutout)
        self.phot_refine_cut = QtWidgets.QCheckBox("Refine star position on preview frame"); self.phot_refine_cut.stateChanged.connect(self.update_phot_cutout)
        form.addRow("Cutout half-size [px]", self.phot_cut_half)
        form.addRow(self.phot_log_cut)
        form.addRow(self.phot_refine_cut)

        left_outer.addLayout(form)

        row_btn = QtWidgets.QHBoxLayout()
        self.phot_preview_btn = QtWidgets.QPushButton("Preview photometry on this frame")
        self.phot_preview_btn.clicked.connect(self.preview_photometry_frame)
        self.phot_run_btn = QtWidgets.QPushButton("Measure photometry (all frames)")
        self.phot_run_btn.clicked.connect(self.run_photometry_all_frames)
        row_btn.addWidget(self.phot_preview_btn)
        row_btn.addWidget(self.phot_run_btn)
        left_outer.addLayout(row_btn)

        save_row = QtWidgets.QHBoxLayout()
        self.phot_save_csv_btn = QtWidgets.QPushButton("Save photometry CSV")
        self.phot_save_csv_btn.clicked.connect(self.save_photometry_csv)
        self.phot_save_json_btn = QtWidgets.QPushButton("Save photometry metadata JSON")
        self.phot_save_json_btn.clicked.connect(self.save_photometry_json)
        save_row.addWidget(self.phot_save_csv_btn)
        save_row.addWidget(self.phot_save_json_btn)
        left_outer.addLayout(save_row)

        self.phot_prog = QtWidgets.QProgressBar(); self.phot_prog.setRange(0, 100)
        self.phot_status = QtWidgets.QLabel("Idle")
        self.phot_status.setObjectName("summaryText")
        self.phot_diag = QtWidgets.QLabel("")
        self.phot_diag.setObjectName("summaryText")
        left_outer.addWidget(self.phot_prog)
        left_outer.addWidget(self.phot_status)
        left_outer.addWidget(self.phot_diag)

        # Right (cutout + plots)
        right = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right)
        self.phot_cutout_image = HoverImageWidget(title="Cutout view")
        self.phot_rel_plot = pg.PlotWidget(title="Differential light curve (normalized)")
        self.phot_raw_plot = pg.PlotWidget(title="Raw fluxes")
        for pw in [self.phot_rel_plot, self.phot_raw_plot]:
            pw.setBackground("w")
            pw.showGrid(x=True, y=True, alpha=0.25)

        right_lay.addWidget(self.phot_cutout_image, stretch=2)
        right_lay.addWidget(self.phot_rel_plot, stretch=1)
        right_lay.addWidget(self.phot_raw_plot, stretch=1)

        root.addWidget(left, stretch=0)
        root.addWidget(right, stretch=1)
        self._add_scroll_tab(tab, "Photometry")

        for w in [self.phot_r_ap, self.phot_r_in, self.phot_r_out, self.phot_bkg_stat, self.phot_bkg_model, self.phot_subpixels,
                  self.phot_comp_mode, self.phot_comp_sigma, self.phot_noise_floor, self.phot_edge_margin, self.phot_saturation]:
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self.update_phot_cutout)
            if hasattr(w, "currentTextChanged"):
                w.currentTextChanged.connect(self.update_phot_cutout)

    def _sync_phot_preview_spin(self, value):
        if self.phot_preview_spin.value() != value:
            self.phot_preview_spin.blockSignals(True)
            self.phot_preview_spin.setValue(value)
            self.phot_preview_spin.blockSignals(False)

    def update_phot_inspect_options(self):
        opts = ["target"] + [k for k in ["comp1", "comp2", "comp3", "comp4"] if self.stars["enabled"][k]]
        current = self.phot_inspect_star.currentText()
        self.phot_inspect_star.blockSignals(True)
        self.phot_inspect_star.clear()
        self.phot_inspect_star.addItems(opts)
        if current in opts:
            self.phot_inspect_star.setCurrentText(current)
        self.phot_inspect_star.blockSignals(False)

    def _get_xy_for_star(self, name, frame_i):
        cube = self.active_cube()
        if cube is None:
            return np.nan, np.nan
        if self.centroids_per_frame is not None and name in self.centroids_per_frame:
            arr = self.centroids_per_frame[name]
            if len(arr) == cube.shape[0]:
                x, y = arr[frame_i]
                if np.isfinite(x) and np.isfinite(y):
                    return float(x), float(y)
        return float(self.stars[name]["x"]), float(self.stars[name]["y"])

    def _cutout(self, img, x, y, half):
        ny, nx = img.shape
        x0 = max(int(np.floor(x)) - half, 0)
        x1 = min(int(np.floor(x)) + half + 1, nx)
        y0 = max(int(np.floor(y)) - half, 0)
        y1 = min(int(np.floor(y)) + half + 1, ny)
        return img[y0:y1, x0:x1], x0, y0

    def _radii_valid(self):
        r_ap = float(self.phot_r_ap.value())
        r_in = float(self.phot_r_in.value())
        r_out = float(self.phot_r_out.value())
        return (r_ap > 0) and (r_in > r_ap) and (r_out > r_in)

    def update_phot_cutout(self):
        cube = self.active_cube()
        if cube is None:
            self.phot_cutout_image.clear()
            return

        if not self._radii_valid():
            self.phot_status.setText("Invalid radii: enforce r_ap > 0 and r_ap < r_in < r_out.")
            return

        idx = int(np.clip(self.phot_preview_slider.value(), 0, cube.shape[0] - 1))
        inspect_star = self.phot_inspect_star.currentText() or "target"
        img0 = cube[idx]

        x0, y0 = self._get_xy_for_star(inspect_star, idx)
        if self.phot_refine_cut.isChecked():
            xr, yr = centroid_2d(img0, x0, y0, half_size=min(30, max(5, int(self.phot_cut_half.value()) // 4)))
            x0, y0 = float(xr), float(yr)

        cut, x_origin, y_origin = self._cutout(img0, x0, y0, int(self.phot_cut_half.value()))
        if cut.size == 0:
            self.phot_cutout_image.clear()
            return

        vmin, vmax = percentile_vmin_vmax(cut)
        disp = cut.copy()
        if self.phot_log_cut.isChecked():
            disp = np.clip(disp, max(vmin, 1e-6), None)
            disp = np.log10(disp)
            vmin, vmax = np.nanpercentile(disp, 5), np.nanpercentile(disp, 99.5)

        self.phot_cutout_image.set_image(disp, vmin=vmin, vmax=vmax)

        xc = float(x0 - x_origin)
        yc = float(y0 - y_origin)
        self.phot_cutout_image.set_points(
            [
                {
                    "name": inspect_star,
                    "x": xc,
                    "y": yc,
                    "symbol": "x",
                    "size": 14,
                    "pen": "#EAB308",
                    "brush": "#EAB308",
                }
            ]
        )
        self.phot_cutout_image.set_circles(
            [
                {"x": xc, "y": yc, "r": float(self.phot_r_ap.value()), "color": "#EF4444", "width": 2.0},
                {"x": xc, "y": yc, "r": float(self.phot_r_in.value()), "color": "#22D3EE", "width": 1.6, "style": QtCore.Qt.DashLine},
                {"x": xc, "y": yc, "r": float(self.phot_r_out.value()), "color": "#22D3EE", "width": 1.6, "style": QtCore.Qt.DashLine},
            ]
        )
        self.phot_cutout_image.plot.getPlotItem().setTitle(f"Cutout around {inspect_star} | frame {idx}")

    def preview_photometry_frame(self):
        if not self._require_data():
            return
        if not self._radii_valid():
            self._show_error("Invalid radii: enforce r_ap > 0 and r_ap < r_in < r_out.")
            return

        cube = self.active_cube()
        idx = int(np.clip(self.phot_preview_slider.value(), 0, cube.shape[0] - 1))
        inspect_star = self.phot_inspect_star.currentText() or "target"
        img0 = cube[idx]
        x0, y0 = self._get_xy_for_star(inspect_star, idx)

        f, ef, bkg, extra = aperture_photometry_fast(
            img0,
            x0,
            y0,
            float(self.phot_r_ap.value()),
            float(self.phot_r_in.value()),
            float(self.phot_r_out.value()),
            bkg_stat=self.phot_bkg_stat.currentText(),
            bkg_model=self.phot_bkg_model.currentText(),
            subpixels=int(self.phot_subpixels.currentText()),
            saturation_level=(float(self.phot_saturation.value()) if self.phot_saturation.value() > 0 else None),
        )

        msg = f"{inspect_star} flux={f:.3g} ADU, err={ef:.3g} ADU, bkg={bkg:.3g} ADU/pix"
        if isinstance(extra, dict) and extra:
            msg += (
                f" | n_ap={extra.get('n_ap')}, n_an={extra.get('n_an')}, "
                f"bkg_std={extra.get('bkg_std')}, peak={extra.get('peak')}"
            )
        self.phot_status.setText(msg)

    def run_photometry_all_frames(self):
        if not self._require_data():
            return
        if not self._radii_valid():
            self._show_error("Invalid radii: enforce r_ap > 0 and r_ap < r_in < r_out.")
            return

        enabled_comps = [k for k in ["comp1", "comp2", "comp3", "comp4"] if self.stars["enabled"][k]]
        if len(enabled_comps) == 0:
            self._show_error("Enable at least one comparison star.")
            return

        cube = self.active_cube()
        n_total = int(cube.shape[0])
        n_process = int(self.phot_max_frames.value()) if int(self.phot_max_frames.value()) > 0 else n_total
        n_process = min(n_process, n_total)

        t = []
        rel = []
        srel = []
        raw_target = []
        raw_ref = []
        raw_target_err = []
        raw_ref_err = []
        airmass = []
        quality_ok = []
        quality_reason = []

        r_ap = float(self.phot_r_ap.value())
        r_in = float(self.phot_r_in.value())
        r_out = float(self.phot_r_out.value())
        bkg_stat = self.phot_bkg_stat.currentText()
        bkg_model = self.phot_bkg_model.currentText()
        subpixels = int(self.phot_subpixels.currentText())
        comp_mode = self.phot_comp_mode.currentText()
        comp_sigma_clip = float(self.phot_comp_sigma.value())
        noise_floor_ppm = float(self.phot_noise_floor.value())
        edge_margin = float(self.phot_edge_margin.value())
        saturation_level = float(self.phot_saturation.value())
        drop_bad_frames = self.phot_drop_bad.isChecked()
        discard_nan_times = self.phot_discard_nan_t.isChecked()

        self._set_busy(True)
        self.phot_prog.setValue(0)
        t0_loop = time.time()

        for i in range(n_total):
            img = cube[i]
            jd = self.images[i].get("jd", np.nan)
            am = self.images[i].get("airmass", np.nan)
            exptime_s = _safe_float(self.images[i].get("exptime", np.nan), np.nan)

            if i >= n_process:
                self.phot_prog.setValue(int(100 * (i + 1) / max(n_total, 1)))
                continue

            if discard_nan_times and (not np.isfinite(jd)):
                self.phot_prog.setValue(int(100 * (i + 1) / max(n_total, 1)))
                continue

            frame_reason = []
            ny, nx = img.shape

            xt, yt = self._get_xy_for_star("target", i)
            if (xt < edge_margin) or (yt < edge_margin) or (xt > (nx - 1 - edge_margin)) or (yt > (ny - 1 - edge_margin)):
                frame_reason.append("target_near_edge")

            ft, eft, _, extra_t = aperture_photometry_fast(
                img,
                xt,
                yt,
                r_ap,
                r_in,
                r_out,
                bkg_stat=bkg_stat,
                bkg_model=bkg_model,
                subpixels=subpixels,
                saturation_level=(saturation_level if saturation_level > 0 else None),
            )
            if isinstance(extra_t, dict) and extra_t.get("is_saturated", False):
                frame_reason.append("target_saturated")

            comp_flux = []
            comp_err = []
            for k in enabled_comps:
                xc, yc = self._get_xy_for_star(k, i)
                if (xc < edge_margin) or (yc < edge_margin) or (xc > (nx - 1 - edge_margin)) or (yc > (ny - 1 - edge_margin)):
                    frame_reason.append(f"{k}_near_edge")
                fc, efc, _, extra_c = aperture_photometry_fast(
                    img,
                    xc,
                    yc,
                    r_ap,
                    r_in,
                    r_out,
                    bkg_stat=bkg_stat,
                    bkg_model=bkg_model,
                    subpixels=subpixels,
                    saturation_level=(saturation_level if saturation_level > 0 else None),
                )
                if isinstance(extra_c, dict) and extra_c.get("is_saturated", False):
                    frame_reason.append(f"{k}_saturated")
                comp_flux.append(fc)
                comp_err.append(efc)

            fref, eref = combine_comps(
                comp_flux,
                comp_err,
                mode=comp_mode,
                sigma_clip=(comp_sigma_clip if comp_sigma_clip > 0 else None),
            )

            if not np.isfinite(ft) or not np.isfinite(eft) or (ft <= 0) or (eft <= 0):
                frame_reason.append("bad_target_flux")
            if not np.isfinite(fref) or not np.isfinite(eref) or (fref <= 0) or (eref <= 0):
                frame_reason.append("bad_comp_flux")

            frame_is_good = len(frame_reason) == 0
            if drop_bad_frames and (not frame_is_good):
                self.phot_prog.setValue(int(100 * (i + 1) / max(n_total, 1)))
                elapsed = time.time() - t0_loop
                eta = (elapsed / max(i + 1, 1)) * max(n_process - (i + 1), 0)
                self.phot_status.setText(f"Processed {i + 1}/{n_process} | elapsed={elapsed:.1f}s eta={eta:.1f}s")
                QtWidgets.QApplication.processEvents()
                continue

            raw_target.append(ft)
            raw_target_err.append(eft)
            raw_ref.append(fref)
            raw_ref_err.append(eref)
            quality_ok.append(frame_is_good)
            quality_reason.append(",".join(frame_reason) if frame_reason else "ok")

            if np.isfinite(ft) and np.isfinite(fref) and (ft > 0) and (fref > 0):
                r = ft / fref
                er = abs(r) * math.sqrt(
                    (eft / max(abs(ft), 1e-12)) ** 2 +
                    (eref / max(abs(fref), 1e-12)) ** 2
                )
                if noise_floor_ppm > 0:
                    er = math.sqrt(er ** 2 + (abs(r) * noise_floor_ppm * 1e-6) ** 2)
            else:
                r, er = np.nan, np.nan

            t.append(jd if np.isfinite(jd) else float(i))
            airmass.append(am)
            rel.append(r)
            srel.append(er)

            self.phot_prog.setValue(int(100 * (i + 1) / max(n_total, 1)))
            elapsed = time.time() - t0_loop
            eta = (elapsed / max(i + 1, 1)) * max(n_process - (i + 1), 0)
            self.phot_status.setText(f"Processed {i + 1}/{n_process} | elapsed={elapsed:.1f}s eta={eta:.1f}s")
            QtWidgets.QApplication.processEvents()

        self._set_busy(False)

        t = np.asarray(t, float)
        rel = np.asarray(rel, float)
        srel = np.asarray(srel, float)
        airmass = np.asarray(airmass, float)
        raw_target = np.asarray(raw_target, float)
        raw_ref = np.asarray(raw_ref, float)
        raw_target_err = np.asarray(raw_target_err, float)
        raw_ref_err = np.asarray(raw_ref_err, float)
        quality_ok = np.asarray(quality_ok, bool)
        quality_reason = np.asarray(quality_reason, object)

        ordr = np.argsort(np.where(np.isfinite(t), t, np.inf))
        t = t[ordr]
        rel = rel[ordr]
        srel = srel[ordr]
        airmass = airmass[ordr]
        raw_target = raw_target[ordr]
        raw_ref = raw_ref[ordr]
        raw_target_err = raw_target_err[ordr]
        raw_ref_err = raw_ref_err[ordr]
        quality_ok = quality_ok[ordr]
        quality_reason = quality_reason[ordr]

        med = np.nanmedian(rel)
        if np.isfinite(med) and med != 0:
            rel = rel / med
            srel = srel / abs(med)

        self.last_phot = dict(
            t=t,
            rel=rel,
            srel=srel,
            airmass=airmass,
            raw_target=raw_target,
            raw_ref=raw_ref,
            raw_target_err=raw_target_err,
            raw_ref_err=raw_ref_err,
            quality_ok=quality_ok,
            quality_reason=quality_reason,
            r_ap=r_ap,
            r_in=r_in,
            r_out=r_out,
            bkg_stat=bkg_stat,
            bkg_model=bkg_model,
            comp_mode=comp_mode,
            comp_sigma_clip=comp_sigma_clip,
            subpixels=subpixels,
            noise_floor_ppm=noise_floor_ppm,
        )

        self.phot_status.setText(f"Photometry done. Stored LAST_PHOT with N={len(t)} points (time-sorted).")
        self.update_phot_plots()
        self.update_detrend_defaults()
        self.update_detrend_plots()
        self.update_mcmc_guess_defaults(force=True)
        self.update_mcmc_preview_plot()

    def _plot_errorbars(self, pw, x, y, yerr, color="#0EA5E9", symbol="o"):
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr >= 0)
        if np.sum(mask) == 0:
            return
        xx = np.asarray(x[mask], float)
        yy = np.asarray(y[mask], float)
        ee = np.asarray(yerr[mask], float)
        pw.addItem(pg.ErrorBarItem(x=xx, y=yy, top=ee, bottom=ee, beam=0.0, pen=pg.mkPen(color, width=1)))
        pw.plot(xx, yy, pen=None, symbol=symbol, symbolSize=6, symbolBrush=color, symbolPen=pg.mkPen(color, width=1))

    def update_phot_plots(self):
        self.phot_rel_plot.clear()
        self.phot_raw_plot.clear()

        if self.last_phot is None:
            self.phot_diag.setText("Run photometry to see the light curve.")
            return

        lp = self.last_phot
        t = np.asarray(lp["t"], float)

        self._plot_errorbars(self.phot_rel_plot, t, np.asarray(lp["rel"], float), np.asarray(lp["srel"], float), color="#0EA5E9")
        self.phot_rel_plot.addItem(pg.InfiniteLine(pos=1.0, angle=0, pen=pg.mkPen((120, 120, 120), style=QtCore.Qt.DashLine)))
        self.phot_rel_plot.setLabel("left", "Relative flux")
        self.phot_rel_plot.setLabel("bottom", "Time (JD)")

        self.phot_raw_plot.plot(t, np.asarray(lp["raw_target"], float), pen=None, symbol="o", symbolSize=6, symbolBrush=STAR_COLORS["target"], symbolPen=pg.mkPen(STAR_COLORS["target"]))
        self.phot_raw_plot.plot(t, np.asarray(lp["raw_ref"], float), pen=None, symbol="o", symbolSize=6, symbolBrush=STAR_COLORS["comp1"], symbolPen=pg.mkPen(STAR_COLORS["comp1"]))
        self.phot_raw_plot.setLabel("left", "Flux [counts]")
        self.phot_raw_plot.setLabel("bottom", "Time (JD)")

        snr = np.nanmedian(np.asarray(lp["raw_target"]) / np.maximum(np.asarray(lp["raw_target_err"]), 1e-12))
        n_bad = int(np.sum(~np.asarray(lp.get("quality_ok", np.ones_like(lp["t"], dtype=bool)), bool)))
        self.phot_diag.setText(f"Median target SNR (rough) ~ {snr:.1f} | Frames flagged during quality checks: {n_bad}")

    def save_photometry_csv(self):
        if self.last_phot is None:
            self._show_error("No photometry to save.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save photometry CSV", "photometry_results.csv", "CSV (*.csv)")
        if not path:
            return

        lp = self.last_phot
        qok = np.asarray(lp.get("quality_ok", np.ones_like(lp["t"], dtype=bool)))
        qrs = np.asarray(lp.get("quality_reason", np.array(["ok"] * len(lp["t"]), dtype=object)))

        with open(path, "w", encoding="utf-8") as f:
            f.write("time_jd,rel_flux,rel_flux_err,airmass,raw_target,raw_ref,target_err,ref_err,quality_ok,quality_reason\n")
            for i in range(len(lp["t"])):
                f.write(
                    f"{lp['t'][i]},{lp['rel'][i]},{lp['srel'][i]},{lp['airmass'][i]},"
                    f"{lp['raw_target'][i]},{lp['raw_ref'][i]},{lp['raw_target_err'][i]},{lp['raw_ref_err'][i]},"
                    f"{bool(qok[i])},{qrs[i]}\n"
                )
        self._show_info(f"Saved CSV: {path}")

    def save_photometry_json(self):
        if self.last_phot is None:
            self._show_error("No photometry metadata to save.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save metadata JSON", "photometry_metadata.json", "JSON (*.json)")
        if not path:
            return

        lp = self.last_phot
        meta = {
            "n_points": int(len(lp["t"])),
            "r_ap": float(lp.get("r_ap", np.nan)),
            "r_in": float(lp.get("r_in", np.nan)),
            "r_out": float(lp.get("r_out", np.nan)),
            "bkg_stat": lp.get("bkg_stat", "median"),
            "bkg_model": lp.get("bkg_model", "annulus"),
            "comp_mode": lp.get("comp_mode", "weighted"),
            "comp_sigma_clip": float(lp.get("comp_sigma_clip", np.nan)),
            "subpixels": int(lp.get("subpixels", 1)),
            "noise_floor_ppm": float(lp.get("noise_floor_ppm", 0.0)),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        self._show_info(f"Saved JSON: {path}")

    # -------------------------
    # Tab: Detrend
    # -------------------------
    def _build_tab_detrend(self):
        tab = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(tab)

        left = QtWidgets.QGroupBox("Detrend controls")
        form = QtWidgets.QFormLayout(left)

        self.det_model = QtWidgets.QComboBox()
        self.det_model.addItems(["none", "time_linear", "time_quadratic", "airmass_linear", "airmass_quadratic"])
        self.det_model.setCurrentText("airmass_linear")
        form.addRow("Detrend against", self.det_model)

        self.det_discard_first = QtWidgets.QSpinBox(); self.det_discard_first.setRange(0, 100000)
        self.det_discard_last = QtWidgets.QSpinBox(); self.det_discard_last.setRange(0, 100000)
        form.addRow("Discard first N points", self.det_discard_first)
        form.addRow("Discard last N points", self.det_discard_last)

        self.det_center_x = QtWidgets.QCheckBox("Center x (stability)")
        self.det_center_x.setChecked(True)
        form.addRow(self.det_center_x)

        self.det_fit_region = QtWidgets.QComboBox()
        self.det_fit_region.addItems(["all kept points", "out-of-transit points only"])
        form.addRow("Fit trend on", self.det_fit_region)

        self.det_t0_oot = QtWidgets.QDoubleSpinBox(); self.det_t0_oot.setDecimals(8); self.det_t0_oot.setRange(-1e9, 1e9)
        self.det_dur_oot = QtWidgets.QDoubleSpinBox(); self.det_dur_oot.setRange(0.0, 100.0); self.det_dur_oot.setValue(0.08)
        form.addRow("Transit center t0 [JD]", self.det_t0_oot)
        form.addRow("Transit duration [days]", self.det_dur_oot)

        self.det_apply_btn = QtWidgets.QPushButton("Apply detrend")
        self.det_apply_btn.clicked.connect(self.apply_detrend)
        form.addRow(self.det_apply_btn)

        self.det_status = QtWidgets.QLabel("Run photometry first.")
        self.det_status.setObjectName("summaryText")
        form.addRow(self.det_status)

        right = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right)
        self.det_plot = pg.PlotWidget(title="Detrend result")
        self.det_trend_plot = pg.PlotWidget(title="Trend model")
        for pw in [self.det_plot, self.det_trend_plot]:
            pw.setBackground("w")
            pw.showGrid(x=True, y=True, alpha=0.25)
            right_lay.addWidget(pw)

        root.addWidget(left, stretch=0)
        root.addWidget(right, stretch=1)
        self._add_scroll_tab(tab, "Detrend")

    def update_detrend_defaults(self):
        if self.last_phot is None:
            return
        t = np.asarray(self.last_phot["t"], float)
        if np.any(np.isfinite(t)):
            self.det_t0_oot.setValue(float(np.nanmedian(t)))

    def apply_detrend(self):
        if self.last_phot is None:
            self._show_error("Run photometry first.")
            return

        lp = self.last_phot
        t = np.asarray(lp["t"], float)
        rel = np.asarray(lp["rel"], float)
        srel = np.asarray(lp["srel"], float)
        am = np.asarray(lp.get("airmass", np.full_like(t, np.nan)), float)

        model = self.det_model.currentText()
        discard_first = int(self.det_discard_first.value())
        discard_last = int(self.det_discard_last.value())
        center_x = self.det_center_x.isChecked()
        fit_region = self.det_fit_region.currentText()
        t0_oot = float(self.det_t0_oot.value())
        dur_oot = float(self.det_dur_oot.value())

        n = len(t)
        df = int(min(discard_first, max(n - 2, 0)))
        dl = int(min(discard_last, max(n - 2 - df, 0)))

        mask = np.ones(n, dtype=bool)
        if df > 0:
            mask[:df] = False
        if dl > 0:
            mask[-dl:] = False

        t_fit = t[mask]
        rel_fit = rel[mask]
        srel_fit = srel[mask]
        am_fit = am[mask]

        warning = ""

        if model == "none":
            rel_d_fit, srel_d_fit, trend_fit = rel_fit, srel_fit, np.ones_like(rel_fit)
        else:
            if model.startswith("airmass"):
                used_x = am_fit
                if np.sum(np.isfinite(used_x)) < max(5, int(0.5 * len(used_x))):
                    warning = "Airmass missing/insufficient; using time instead."
                    used_x = t_fit
            else:
                used_x = t_fit

            degree = 1 if "linear" in model else 2
            fit_mask = np.ones_like(rel_fit, dtype=bool)
            if fit_region.startswith("out-of-transit"):
                fit_mask = np.abs(t_fit - t0_oot) > (0.5 * dur_oot)
                if np.sum(fit_mask) < (degree + 1):
                    warning = "OOT mask too small; using all kept points."
                    fit_mask = np.ones_like(rel_fit, dtype=bool)

            rel_d_fit, srel_d_fit, trend_fit = detrend_flux(
                rel_fit,
                srel_fit,
                used_x,
                degree=degree,
                center_x=center_x,
                fit_mask=fit_mask,
            )

            med = np.nanmedian(rel_d_fit)
            if np.isfinite(med) and med != 0:
                rel_d_fit = rel_d_fit / med
                srel_d_fit = srel_d_fit / abs(med)

        rel_d = np.full_like(rel, np.nan, dtype=float)
        srel_d = np.full_like(srel, np.nan, dtype=float)
        trend = np.full_like(rel, np.nan, dtype=float)
        rel_d[mask] = rel_d_fit
        srel_d[mask] = srel_d_fit
        trend[mask] = trend_fit

        lp["discard_mask"] = mask
        lp["rel_detrended"] = rel_d
        lp["srel_detrended"] = srel_d
        lp["trend_model"] = trend
        lp["detrend_model"] = model
        lp["detrend_fit_region"] = fit_region
        self.last_phot = lp

        self.det_status.setText("Detrending applied. " + warning if warning else "Detrending applied.")
        self.update_detrend_plots()
        self.update_mcmc_preview_plot()

    def update_detrend_plots(self):
        self.det_plot.clear()
        self.det_trend_plot.clear()

        if self.last_phot is None:
            self.det_status.setText("Run photometry first.")
            return

        lp = self.last_phot
        t = np.asarray(lp["t"], float)
        rel = np.asarray(lp["rel"], float)
        rel_d = lp.get("rel_detrended", None)
        mask = np.asarray(lp.get("discard_mask", np.ones_like(t, dtype=bool)), bool)

        self.det_plot.plot(t, rel, pen=None, symbol="o", symbolSize=5, symbolBrush=(140, 140, 140, 120), symbolPen=(140, 140, 140, 120))
        self.det_plot.plot(t[mask], rel[mask], pen=None, symbol="o", symbolSize=6, symbolBrush="#0EA5E9", symbolPen="#0EA5E9")
        if rel_d is not None:
            rel_d = np.asarray(rel_d, float)
            self.det_plot.plot(t[mask], rel_d[mask], pen=None, symbol="o", symbolSize=6, symbolBrush="#10B981", symbolPen="#10B981")
        self.det_plot.addItem(pg.InfiniteLine(pos=1.0, angle=0, pen=pg.mkPen((120, 120, 120), style=QtCore.Qt.DashLine)))
        self.det_plot.setLabel("left", "Relative flux")
        self.det_plot.setLabel("bottom", "Time (JD)")

        if "trend_model" in lp:
            tr = np.asarray(lp["trend_model"], float)
            self.det_trend_plot.plot(t[mask], tr[mask], pen=pg.mkPen("#F59E0B", width=2), symbol="o", symbolSize=5, symbolBrush="#F59E0B")
            self.det_trend_plot.setLabel("left", "Trend")
            self.det_trend_plot.setLabel("bottom", "Time (JD)")

    # -------------------------
    # Tab: Fit (Batman or Polynomial)
    # -------------------------
    def _build_tab_mcmc(self):
        tab = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(tab)

        left = QtWidgets.QGroupBox("Fit controls")
        left_col = QtWidgets.QVBoxLayout(left)

        top_form = QtWidgets.QFormLayout()
        self.mcmc_data_choice = QtWidgets.QComboBox()
        self.mcmc_data_choice.addItems(["raw (kept mask if any)", "detrended (kept mask if any)"])
        self.mcmc_data_choice.currentTextChanged.connect(self.update_mcmc_guess_defaults)
        self.mcmc_data_choice.currentTextChanged.connect(self.update_mcmc_preview_plot)
        self.mcmc_data_choice.currentTextChanged.connect(self.update_mcmc_result_plot)
        top_form.addRow("Fit which data", self.mcmc_data_choice)

        self.fit_mode_combo = QtWidgets.QComboBox()
        self.fit_mode_combo.addItems(["Batman transit (MCMC)", "Polynomial"])
        self.fit_mode_combo.currentTextChanged.connect(self._on_fit_mode_changed)
        top_form.addRow("Fit model", self.fit_mode_combo)
        left_col.addLayout(top_form)

        self.batman_box = QtWidgets.QGroupBox("Batman transit controls")
        bform = QtWidgets.QFormLayout(self.batman_box)

        self.mcmc_P = QtWidgets.QDoubleSpinBox(); self.mcmc_P.setRange(1e-6, 1e6); self.mcmc_P.setValue(1.0); self.mcmc_P.setDecimals(8)
        self.mcmc_ecc = QtWidgets.QDoubleSpinBox(); self.mcmc_ecc.setRange(0.0, 0.99); self.mcmc_ecc.setValue(0.0)
        self.mcmc_w = QtWidgets.QDoubleSpinBox(); self.mcmc_w.setRange(0.0, 360.0); self.mcmc_w.setValue(90.0)
        self.mcmc_u1 = QtWidgets.QDoubleSpinBox(); self.mcmc_u1.setRange(-1.0, 2.0); self.mcmc_u1.setValue(0.3)
        self.mcmc_u2 = QtWidgets.QDoubleSpinBox(); self.mcmc_u2.setRange(-1.0, 2.0); self.mcmc_u2.setValue(0.2)
        bform.addRow("Period P [days]", self.mcmc_P)
        bform.addRow("Eccentricity", self.mcmc_ecc)
        bform.addRow("omega [deg]", self.mcmc_w)
        bform.addRow("Limb darkening u1", self.mcmc_u1)
        bform.addRow("Limb darkening u2", self.mcmc_u2)

        self.mcmc_t0 = QtWidgets.QDoubleSpinBox(); self.mcmc_t0.setDecimals(8); self.mcmc_t0.setRange(-1e9, 1e9)
        self.mcmc_rp = QtWidgets.QDoubleSpinBox(); self.mcmc_rp.setRange(0.001, 0.5); self.mcmc_rp.setValue(0.1); self.mcmc_rp.setSingleStep(0.001); self.mcmc_rp.setDecimals(5)
        self.mcmc_a = QtWidgets.QDoubleSpinBox(); self.mcmc_a.setRange(1.0, 200.0); self.mcmc_a.setValue(10.0)
        self.mcmc_inc = QtWidgets.QDoubleSpinBox(); self.mcmc_inc.setRange(60.0, 90.0); self.mcmc_inc.setValue(87.0); self.mcmc_inc.setSingleStep(0.1)
        self.mcmc_baseline = QtWidgets.QDoubleSpinBox(); self.mcmc_baseline.setRange(0.5, 1.5); self.mcmc_baseline.setValue(1.0); self.mcmc_baseline.setDecimals(8)
        self.mcmc_jitter = QtWidgets.QDoubleSpinBox(); self.mcmc_jitter.setRange(1e-9, 1.0); self.mcmc_jitter.setValue(1e-3); self.mcmc_jitter.setDecimals(8)
        bform.addRow("t0 guess [JD]", self.mcmc_t0)
        bform.addRow("rp = Rp/Rs", self.mcmc_rp)
        bform.addRow("a/Rs", self.mcmc_a)
        bform.addRow("inc [deg]", self.mcmc_inc)
        bform.addRow("baseline", self.mcmc_baseline)
        bform.addRow("jitter", self.mcmc_jitter)
        self.mcmc_rstar = QtWidgets.QDoubleSpinBox(); self.mcmc_rstar.setRange(0.0, 1000.0); self.mcmc_rstar.setValue(0.0); self.mcmc_rstar.setSingleStep(0.01); self.mcmc_rstar.setDecimals(6)
        bform.addRow("Star radius R* [Rsun] (0=unknown)", self.mcmc_rstar)

        self.mcmc_walkers = QtWidgets.QSpinBox(); self.mcmc_walkers.setRange(12, 1024); self.mcmc_walkers.setValue(64)
        self.mcmc_burn = QtWidgets.QSpinBox(); self.mcmc_burn.setRange(10, 100000); self.mcmc_burn.setValue(1000)
        self.mcmc_prod = QtWidgets.QSpinBox(); self.mcmc_prod.setRange(10, 100000); self.mcmc_prod.setValue(1500)
        bform.addRow("Walkers", self.mcmc_walkers)
        bform.addRow("Burn steps", self.mcmc_burn)
        bform.addRow("Prod steps", self.mcmc_prod)

        self.mcmc_show_corner = QtWidgets.QCheckBox("Show corner plot")
        bform.addRow(self.mcmc_show_corner)
        self.mcmc_run_btn = QtWidgets.QPushButton("Run Batman MCMC")
        self.mcmc_run_btn.clicked.connect(self.run_mcmc)
        bform.addRow(self.mcmc_run_btn)

        self.poly_box = QtWidgets.QGroupBox("Polynomial fit controls")
        pform = QtWidgets.QFormLayout(self.poly_box)
        self.poly_degree = QtWidgets.QSpinBox(); self.poly_degree.setRange(1, 10); self.poly_degree.setValue(2)
        self.poly_center_x = QtWidgets.QCheckBox("Center x (stability)"); self.poly_center_x.setChecked(True)
        self.poly_x_choice = QtWidgets.QComboBox(); self.poly_x_choice.addItems(["time", "airmass"])
        pform.addRow("Polynomial degree", self.poly_degree)
        pform.addRow("Polynomial x variable", self.poly_x_choice)
        pform.addRow(self.poly_center_x)
        self.poly_run_btn = QtWidgets.QPushButton("Run polynomial fit")
        self.poly_run_btn.clicked.connect(self.run_polynomial_fit)
        pform.addRow(self.poly_run_btn)

        self.poly_degree.valueChanged.connect(self.update_mcmc_preview_plot)
        self.poly_x_choice.currentTextChanged.connect(self.update_mcmc_preview_plot)
        self.poly_center_x.stateChanged.connect(self.update_mcmc_preview_plot)

        left_col.addWidget(self.batman_box)
        left_col.addWidget(self.poly_box)

        self.mcmc_status = QtWidgets.QLabel("Run photometry first.")
        self.mcmc_status.setObjectName("summaryText")
        left_col.addWidget(self.mcmc_status)
        left_col.addStretch(1)

        right = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right)
        self.mcmc_preview_plot = pg.PlotWidget(title="Preview fit model")
        self.mcmc_result_plot = pg.PlotWidget(title="Fit result")
        for pw in [self.mcmc_preview_plot, self.mcmc_result_plot]:
            pw.setBackground("w")
            pw.showGrid(x=True, y=True, alpha=0.25)
        self.mcmc_summary_box = QtWidgets.QPlainTextEdit()
        self.mcmc_summary_box.setReadOnly(True)
        self.mcmc_summary_box.setPlainText("No fit result yet.")
        right_lay.addWidget(self.mcmc_preview_plot, stretch=1)
        right_lay.addWidget(self.mcmc_result_plot, stretch=1)
        right_lay.addWidget(self.mcmc_summary_box, stretch=1)

        root.addWidget(left, stretch=0)
        root.addWidget(right, stretch=1)
        self._add_scroll_tab(tab, "Fit")

        for w in [self.mcmc_P, self.mcmc_ecc, self.mcmc_w, self.mcmc_u1, self.mcmc_u2,
                  self.mcmc_t0, self.mcmc_rp, self.mcmc_a, self.mcmc_inc, self.mcmc_baseline, self.mcmc_jitter]:
            w.valueChanged.connect(self.update_mcmc_preview_plot)
        self.mcmc_rstar.valueChanged.connect(self.update_mcmc_result_plot)

        self._on_fit_mode_changed()

    def _fit_mode_is_batman(self):
        return self.fit_mode_combo.currentText().startswith("Batman")

    def _on_fit_mode_changed(self):
        is_batman = self._fit_mode_is_batman()
        self.batman_box.setVisible(is_batman)
        self.poly_box.setVisible(not is_batman)
        self.update_mcmc_guess_defaults(force=False)
        self.update_mcmc_preview_plot()
        self.update_mcmc_result_plot()

    def _current_fit_data_with_mask(self):
        if self.last_phot is None:
            return None, None, None, "", None

        lp = self.last_phot
        t = np.asarray(lp["t"], float)
        mask = np.asarray(lp.get("discard_mask", np.ones_like(t, dtype=bool)), bool)

        if self.mcmc_data_choice.currentText().startswith("detrended") and "rel_detrended" in lp and "srel_detrended" in lp:
            y = np.asarray(lp["rel_detrended"], float)
            yerr = np.asarray(lp["srel_detrended"], float)
            used = "detrended"
        else:
            y = np.asarray(lp["rel"], float)
            yerr = np.asarray(lp["srel"], float)
            used = "raw"

        mm = mask & np.isfinite(t) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
        return t[mm], y[mm], yerr[mm], used, mm

    def _current_mcmc_data(self):
        t_fit, y_fit, yerr_fit, used, _ = self._current_fit_data_with_mask()
        return t_fit, y_fit, yerr_fit, used

    def _compute_polynomial_fit(self):
        t_fit, y_fit, yerr_fit, used, mm = self._current_fit_data_with_mask()
        if t_fit is None or len(t_fit) == 0:
            return None

        x_mode = self.poly_x_choice.currentText()
        fallback_to_time = False
        if x_mode == "airmass":
            am = np.asarray(self.last_phot.get("airmass", np.full_like(self.last_phot["t"], np.nan)), float)
            x_fit = am[mm]
            if np.sum(np.isfinite(x_fit)) < max(5, int(0.5 * len(x_fit))):
                x_fit = t_fit.copy()
                x_label = "Time (JD)"
                fallback_to_time = True
            else:
                x_label = "Airmass"
        else:
            x_fit = t_fit.copy()
            x_label = "Time (JD)"

        x_work = np.asarray(x_fit, float)
        center_x = self.poly_center_x.isChecked()
        if center_x:
            x_work = x_work - np.nanmedian(x_work)

        degree = int(self.poly_degree.value())
        coeffs, y_model, _ = weighted_polyfit(x_work, y_fit, yerr_fit, degree=degree)
        if coeffs is None:
            return None

        good = np.isfinite(y_model) & np.isfinite(y_fit) & np.isfinite(yerr_fit) & (yerr_fit > 0)
        if np.sum(good) == 0:
            return None
        resid = y_fit[good] - y_model[good]
        chi2 = float(np.sum((resid / np.maximum(yerr_fit[good], 1e-12)) ** 2))
        dof = int(max(np.sum(good) - (degree + 1), 1))
        redchi2 = float(chi2 / dof)
        rms = float(np.sqrt(np.nanmean(resid ** 2)))

        return dict(
            t=t_fit,
            y=y_fit,
            yerr=yerr_fit,
            x=x_fit,
            x_label=x_label,
            used=used,
            degree=degree,
            center_x=center_x,
            coeffs=np.asarray(coeffs, float),
            y_model=np.asarray(y_model, float),
            chi2=chi2,
            dof=dof,
            redchi2=redchi2,
            rms=rms,
            rms_ppm=float(rms * 1e6),
            fallback_to_time=fallback_to_time,
        )

    def update_mcmc_guess_defaults(self, force=False):
        t_fit, y_fit, yerr_fit, used = self._current_mcmc_data()
        if t_fit is None or len(t_fit) == 0:
            return

        if not self._fit_mode_is_batman():
            self.mcmc_status.setText(f"Using {used} data | N={len(t_fit)} points")
            return

        if (not force) and self._manual_mcmc_guess:
            return

        t0_guess = float(t_fit[np.argmin(y_fit)])
        depth_guess = float(np.clip(np.nanmedian(y_fit) - np.nanmin(y_fit), 1e-4, 0.05))
        rp_guess = float(np.sqrt(depth_guess))
        baseline_guess = float(np.nanmedian(y_fit))
        jitter_guess = float(np.nanmedian(yerr_fit) * 0.5)

        for w in [self.mcmc_t0, self.mcmc_rp, self.mcmc_baseline, self.mcmc_jitter]:
            w.blockSignals(True)
        self.mcmc_t0.setValue(t0_guess)
        self.mcmc_rp.setValue(rp_guess)
        self.mcmc_baseline.setValue(baseline_guess)
        self.mcmc_jitter.setValue(max(jitter_guess, 1e-8))
        for w in [self.mcmc_t0, self.mcmc_rp, self.mcmc_baseline, self.mcmc_jitter]:
            w.blockSignals(False)

        self.mcmc_status.setText(f"Using {used} data | N={len(t_fit)} points")

    def update_mcmc_preview_plot(self):
        self.mcmc_preview_plot.clear()

        t_fit, y_fit, yerr_fit, used = self._current_mcmc_data()
        if t_fit is None or len(t_fit) == 0:
            self.mcmc_status.setText("No valid points for selected data.")
            return

        self._plot_errorbars(self.mcmc_preview_plot, t_fit, y_fit, yerr_fit, color="#0EA5E9")
        self.mcmc_preview_plot.addItem(pg.InfiniteLine(pos=1.0, angle=0, pen=pg.mkPen((120, 120, 120), style=QtCore.Qt.DashLine)))
        self.mcmc_preview_plot.setLabel("left", "Relative flux")
        self.mcmc_preview_plot.setLabel("bottom", "Time (JD)")

        if self._fit_mode_is_batman():
            t_med = np.median(t_fit)
            t_rel = t_fit - t_med
            t0_rel = float(self.mcmc_t0.value() - t_med)

            try:
                model_prev = batman_flux_model(
                    t_rel,
                    t0_rel,
                    float(self.mcmc_rp.value()),
                    float(self.mcmc_a.value()),
                    float(self.mcmc_inc.value()),
                    float(self.mcmc_baseline.value()),
                    float(self.mcmc_P.value()),
                    float(self.mcmc_ecc.value()),
                    float(self.mcmc_w.value()),
                    float(self.mcmc_u1.value()),
                    float(self.mcmc_u2.value()),
                ) if HAS_BATMAN else np.full_like(t_fit, np.nan)
            except Exception:
                model_prev = np.full_like(t_fit, np.nan)

            if np.all(np.isfinite(model_prev)):
                self.mcmc_preview_plot.plot(t_fit, model_prev, pen=pg.mkPen("#111827", width=2))

            msg = f"Using {used} | N={len(t_fit)} | Batman preview"
            if len(t_fit) < 20:
                msg += " | Few points for stable posteriors"
            self.mcmc_status.setText(msg)
            return

        poly = self._compute_polynomial_fit()
        if poly is None:
            self.mcmc_status.setText("Polynomial preview failed: not enough valid points.")
            return

        t_ord = np.argsort(poly["t"])
        self.mcmc_preview_plot.plot(
            poly["t"][t_ord],
            poly["y_model"][t_ord],
            pen=pg.mkPen("#111827", width=2),
        )
        msg = f"Using {poly['used']} | N={len(poly['t'])} | Polynomial degree {poly['degree']} on {poly['x_label']}"
        if poly["fallback_to_time"]:
            msg += " (airmass unavailable -> time)"
        self.mcmc_status.setText(msg)

    def run_polynomial_fit(self):
        if self.last_phot is None:
            self._show_error("Run photometry first.")
            return

        poly = self._compute_polynomial_fit()
        if poly is None:
            self._show_error("Polynomial fit failed. Check data quality and selected degree.")
            return

        self.last_polyfit = poly
        self.mcmc_status.setText("Polynomial fit complete.")
        self.update_mcmc_result_plot()

    def run_mcmc(self):
        if not self._fit_mode_is_batman():
            self.run_polynomial_fit()
            return

        if self.last_phot is None:
            self._show_error("Run photometry first.")
            return
        if not HAS_BATMAN:
            self._show_error("batman-package not installed.")
            return
        if not HAS_EMCEE:
            self._show_error("emcee not installed.")
            return

        t_fit, y_fit, yerr_fit, used = self._current_mcmc_data()
        if t_fit is None or len(t_fit) < 8:
            self._show_error("Need at least 8 valid points for MCMC.")
            return

        walkers = int(self.mcmc_walkers.value())
        if walkers < 12:
            self._show_error("Use at least 12 walkers.")
            return

        fixed = dict(
            P=float(self.mcmc_P.value()),
            ecc=float(self.mcmc_ecc.value()),
            w=float(self.mcmc_w.value()),
            u1=float(self.mcmc_u1.value()),
            u2=float(self.mcmc_u2.value()),
        )
        guesses = dict(
            t0=float(self.mcmc_t0.value()),
            rp=float(self.mcmc_rp.value()),
            a=float(self.mcmc_a.value()),
            inc=float(self.mcmc_inc.value()),
            baseline=float(self.mcmc_baseline.value()),
            jitter=float(self.mcmc_jitter.value()),
        )

        self._set_busy(True)
        self.mcmc_status.setText("Running Batman MCMC...")
        QtWidgets.QApplication.processEvents()

        try:
            res = run_batman_mcmc(
                t_fit,
                y_fit,
                yerr_fit,
                guesses,
                fixed,
                walkers=walkers,
                burn=int(self.mcmc_burn.value()),
                prod=int(self.mcmc_prod.value()),
            )
        except Exception as exc:
            self._set_busy(False)
            self._show_error(exc)
            return

        self._set_busy(False)
        self.last_mcmc = res
        self.mcmc_status.setText("Batman MCMC complete.")
        self.update_mcmc_result_plot()

        if self.mcmc_show_corner.isChecked() and HAS_CORNER:
            samples_abs = np.asarray(res["samples_abs"], float)
            labels = ["T0", "Rp/Rs", "a/Rs", "inc", "baseline", "ln_jitter"]
            q50 = np.percentile(samples_abs, 50, axis=0)
            fig = corner.corner(samples_abs, labels=labels, truths=q50)
            dlg = CornerDialog(fig, self)
            dlg.exec()
            plt.close(fig)
        elif self.mcmc_show_corner.isChecked() and (not HAS_CORNER):
            self._show_info("Install corner to use corner plots: pip install corner")

    def update_mcmc_result_plot(self):
        self.mcmc_result_plot.clear()

        if self._fit_mode_is_batman():
            if self.last_mcmc is None:
                self.mcmc_summary_box.setPlainText("No Batman fit result yet.")
                return

            t_fit, y_fit, yerr_fit, used = self._current_mcmc_data()
            if t_fit is None or len(t_fit) == 0:
                self.mcmc_summary_box.setPlainText("Batman result exists but current data selection has no valid points.")
                return

            res = self.last_mcmc
            samples = np.asarray(res["samples"], float)
            fixed = dict(res["fixed"])

            t_med = np.median(t_fit)
            t_grid = np.linspace(np.min(t_fit), np.max(t_fit), 800)
            t_grid_rel = t_grid - t_med

            rng = np.random.default_rng(0)
            n_draw = min(600, samples.shape[0])
            draw = rng.choice(samples.shape[0], size=n_draw, replace=False)
            models = np.empty((n_draw, t_grid.size), dtype=float)
            for j, idx in enumerate(draw):
                th = samples[idx]
                models[j] = batman_flux_model(
                    t_grid_rel,
                    th[0],
                    th[1],
                    th[2],
                    th[3],
                    th[4],
                    fixed["P"],
                    fixed["ecc"],
                    fixed["w"],
                    fixed["u1"],
                    fixed["u2"],
                )

            m_med = np.median(models, axis=0)
            m_lo = np.percentile(models, 16, axis=0)
            m_hi = np.percentile(models, 84, axis=0)

            self._plot_errorbars(self.mcmc_result_plot, t_fit, y_fit, yerr_fit, color="#0EA5E9")
            lo_item = self.mcmc_result_plot.plot(t_grid, m_lo, pen=pg.mkPen((14, 165, 233, 90), width=1))
            hi_item = self.mcmc_result_plot.plot(t_grid, m_hi, pen=pg.mkPen((14, 165, 233, 90), width=1))
            fill = pg.FillBetweenItem(lo_item, hi_item, brush=pg.mkBrush(14, 165, 233, 50))
            self.mcmc_result_plot.addItem(fill)
            self.mcmc_result_plot.plot(t_grid, m_med, pen=pg.mkPen("#111827", width=2))
            self.mcmc_result_plot.addItem(pg.InfiniteLine(pos=1.0, angle=0, pen=pg.mkPen((120, 120, 120), style=QtCore.Qt.DashLine)))
            self.mcmc_result_plot.setLabel("left", "Relative flux")
            self.mcmc_result_plot.setLabel("bottom", "Time (JD)")

            samples_abs = np.asarray(res["samples_abs"], float)
            labels = ["T0", "Rp/Rs", "a/Rs", "inc", "baseline", "ln_jitter"]
            q16, q50, q84 = np.percentile(samples_abs, [16, 50, 84], axis=0)
            oc_min = float((q50[0] - self.mcmc_t0.value()) * 1440.0)

            lines = []
            lines.append(f"Diagnostics: acceptance={res.get('acceptance_fraction', np.nan):.3f}, tau_med={np.nanmedian(res.get('autocorr_time', np.array([np.nan]))):.2f}, n_eff~{res.get('n_eff', np.nan):.1f}")
            lines.append(f"Data used: {used} | N={len(t_fit)}")
            lines.append(f"O-C [minutes] (vs t0 guess) = {oc_min:+.3f}")
            lines.append("")
            lines.append("Posterior summary (median -1sigma/+1sigma)")
            for i, lab in enumerate(labels):
                lo = q50[i] - q16[i]
                hi = q84[i] - q50[i]
                lines.append(f"{lab}: {q50[i]:.8g}  (-{lo:.3g}, +{hi:.3g})")

            rstar_rsun = float(self.mcmc_rstar.value())
            if np.isfinite(rstar_rsun) and (rstar_rsun > 0):
                rp_rsun = samples_abs[:, 1] * rstar_rsun
                a_rsun = samples_abs[:, 2] * rstar_rsun
                rp_rjup = rp_rsun * RSUN_TO_RJUP
                rp_rearth = rp_rsun * RSUN_TO_REARTH
                a_au = a_rsun * RSUN_TO_AU

                rp_rj_q16, rp_rj_q50, rp_rj_q84 = np.percentile(rp_rjup, [16, 50, 84])
                rp_re_q16, rp_re_q50, rp_re_q84 = np.percentile(rp_rearth, [16, 50, 84])
                a_au_q16, a_au_q50, a_au_q84 = np.percentile(a_au, [16, 50, 84])
                a_rs_q16, a_rs_q50, a_rs_q84 = np.percentile(a_rsun, [16, 50, 84])

                lines.append("")
                lines.append(f"Derived physical values using R*={rstar_rsun:.6g} Rsun")
                lines.append(
                    f"Rp = {rp_rj_q50:.6g} Rjup  (-{rp_rj_q50-rp_rj_q16:.3g}, +{rp_rj_q84-rp_rj_q50:.3g})"
                )
                lines.append(
                    f"Rp = {rp_re_q50:.6g} Rearth  (-{rp_re_q50-rp_re_q16:.3g}, +{rp_re_q84-rp_re_q50:.3g})"
                )
                lines.append(
                    f"a = {a_au_q50:.6g} AU  (-{a_au_q50-a_au_q16:.3g}, +{a_au_q84-a_au_q50:.3g})"
                )
                lines.append(
                    f"a = {a_rs_q50:.6g} Rsun  (-{a_rs_q50-a_rs_q16:.3g}, +{a_rs_q84-a_rs_q50:.3g})"
                )
            else:
                lines.append("")
                lines.append("Set Star radius R* [Rsun] > 0 to output physical Rp and a.")
            self.mcmc_summary_box.setPlainText("\n".join(lines))
            return

        if self.last_polyfit is None:
            self.mcmc_summary_box.setPlainText("No polynomial fit result yet.")
            return

        poly = self.last_polyfit
        t_fit = np.asarray(poly["t"], float)
        y_fit = np.asarray(poly["y"], float)
        yerr_fit = np.asarray(poly["yerr"], float)
        y_model = np.asarray(poly["y_model"], float)

        self._plot_errorbars(self.mcmc_result_plot, t_fit, y_fit, yerr_fit, color="#0EA5E9")
        t_ord = np.argsort(t_fit)
        self.mcmc_result_plot.plot(t_fit[t_ord], y_model[t_ord], pen=pg.mkPen("#111827", width=2))
        self.mcmc_result_plot.addItem(pg.InfiniteLine(pos=1.0, angle=0, pen=pg.mkPen((120, 120, 120), style=QtCore.Qt.DashLine)))
        self.mcmc_result_plot.setLabel("left", "Relative flux")
        self.mcmc_result_plot.setLabel("bottom", "Time (JD)")

        lines = []
        lines.append("Polynomial fit summary")
        lines.append(f"Data used: {poly['used']} | N={len(t_fit)}")
        lines.append(f"Degree: {poly['degree']}")
        lines.append(f"x variable: {poly['x_label']} | center_x={poly['center_x']}")
        if poly.get("fallback_to_time", False):
            lines.append("Airmass not sufficient -> fallback to time")
        lines.append(f"chi2={poly['chi2']:.4g}, dof={poly['dof']}, reduced chi2={poly['redchi2']:.4g}")
        lines.append(f"RMS residual={poly['rms']:.6g} ({poly['rms_ppm']:.1f} ppm)")
        lines.append("")
        lines.append("Coefficients (c0 + c1*x + c2*x^2 + ...)")
        for i, c in enumerate(np.asarray(poly["coeffs"], float)):
            lines.append(f"c{i} = {c:.10g}")
        self.mcmc_summary_box.setPlainText("\n".join(lines))

    # -------------------------
    # Tab switching updates
    # -------------------------
    def _on_tab_changed(self, idx):
        name = self.tabs.tabText(idx)
        if name == "Inspect frames":
            self.update_inspect_view()
        elif name == "Align frames":
            self.update_alignment_plots()
        elif name == "Pick stars":
            self.update_stars_view()
        elif name == "Photometry":
            self.update_phot_inspect_options()
            self.update_phot_cutout()
            self.update_phot_plots()
        elif name == "Detrend":
            self.update_detrend_defaults()
            self.update_detrend_plots()
        elif name in ("Fit", "MCMC"):
            self.update_mcmc_guess_defaults(force=False)
            self.update_mcmc_preview_plot()
            self.update_mcmc_result_plot()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "#0F172A")
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

    win = ExoTransitMainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

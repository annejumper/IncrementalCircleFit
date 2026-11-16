#!/usr/bin/env python3
"""
Interactive, incremental circle-fit modal analysis of ObieApp "acquire" data.

You pick the modes one at a time:
  * The top panel shows rms |H| over all points: measured (grey), the current
    residual (blue) and the sum of the modes accepted so far (orange).
  * Click on a peak (or press  n  for the lowest peak above the last mode).
    The click snaps to the nearest residual maximum, and that peak is
    circle-fitted. The bottom panels show the Nyquist circles for the
    strongest points, the residual before and after removing the mode,
    and the mode shape.
  * a / Enter  accept: the mode is subtracted from every FRF, all frequencies
    x / Esc    discard the proposed fit
    [ / ]      narrow / widen the fit band (one bin each side) and refit
    u          undo the last accepted mode
    <- / ->    (or the Prev / Next buttons) step back and forth through the
               accepted modes: step n shows the residual after n modes and
               the circle fit of mode n.  "Latest" jumps back to the end.
               Clicking a peak or pressing n while browsing returns to the
               latest step first.
    s          save results            q   save and quit
  * Use the toolbar's zoom/pan to look around. Clicks are ignored while
    zoom or pan is active.

How one mode is fitted:
  For each measurement point, a circle is fitted to the receptance (H/(-w^2))
  Nyquist data in the band. The angle-vs-frequency relation on that circle,
  phi = phi0 +/- 2*atan((1 - w^2/wr^2)/eta), gives f_r and the loss factor eta.
  The points are combined (weighted median, weighted by circle size). Unless
  --no-refine is given, f_r and eta are then refined by one least-squares fit
  of the same circle model, A_p/(wr^2 - w^2 + i*eta*wr^2) + B_p, over all
  points at once. B_p is the circle offset that the other modes cause and is
  not subtracted. The mode's contribution -w^2*A_p/(wr^2 - w^2 + i*eta*wr^2)
  is removed from every FRF.

Replay / batch:
  --modes 34.3,73.6,196:4   pre-fit these peaks (Hz, optional :half-width in
  bins) before the GUI opens. The peak_Hz and half_width_bins columns in
  modes.csv are in this format, so you can replay a saved session.
  --no-gui                  just fit --modes and save.

ObieApp .trf file layout (reverse-engineered, little-endian):
  0x26  float64  df (Hz)
  0x36  float64  f_max (Hz)
  0x42  float32  number of frequency bins N
  110   N x (float64 re, float64 im, float64 coherence)
  then  "\\0OBIE_META\\n..." text trailer

Usage:
  python3 circle_fit_modes.py Demo/Demo_07
  python3 circle_fit_modes.py Demo/Demo_08 --fmax 2000
"""
import argparse
import csv
import glob
import json
import os
import re
import struct
import sys

import numpy as np
from scipy.optimize import least_squares

import matplotlib

HEADER_BYTES = 110
MAX_ETA = 0.2      # upper bound on loss factor (avoids circle/offset degeneracy)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def read_trf(path):
    b = open(path, "rb").read()
    df = struct.unpack("<d", b[0x26:0x2E])[0]
    n = int(struct.unpack("<f", b[0x42:0x46])[0])
    if not (0 < df < 100) or n <= 0 or len(b) < HEADER_BYTES + 24 * n:
        raise ValueError(f"{path}: unrecognised .trf header (df={df}, n={n})")
    a = np.frombuffer(b[HEADER_BYTES:HEADER_BYTES + 24 * n], "<f8").reshape(n, 3)
    return df, a[:, 0] + 1j * a[:, 1], a[:, 2]


def load_dataset(folder):
    folder = os.path.abspath(folder)
    tpl_path = os.path.join(folder, "template.json")
    tpl = json.load(open(tpl_path)) if os.path.exists(tpl_path) else {}
    prefix = tpl.get("prefix", "H")
    # Exact "<name> H01.trf" files only; ObieApp's "H11 2.trf" retake
    # leftovers and .crswap temp files are ignored.
    pat = re.compile(r" " + re.escape(prefix) + r"(\d+)\.trf$")
    files = sorted((int(m.group(1)), f)
                   for f in glob.glob(os.path.join(folder, "TRF", "*.trf"))
                   for m in [pat.search(f)] if m)
    if not files:
        sys.exit(f"No {prefix}##.trf files found in {folder}/TRF")

    H, C, df0 = [], [], None
    for _, f in files:
        df, h, c = read_trf(f)
        if df0 is None:
            df0, n0 = df, len(h)
        elif abs(df - df0) > 1e-9 or len(h) != n0:
            sys.exit(f"{f}: frequency axis differs from the other files")
        H.append(h)
        C.append(c)
    labels = [k for k, _ in files]

    xy = None
    nodes = tpl.get("stencil", {}).get("nodes", [])
    if nodes:
        pos = {int(n["label"]): (n["xMm"], n["yMm"]) for n in nodes}
        if all(k in pos for k in labels):
            xy = np.array([pos[k] for k in labels], float)
    return dict(name=os.path.basename(folder), f=np.arange(n0) * df0, df=df0,
                H=np.array(H), coh=np.array(C), labels=labels, xy=xy)


# ----------------------------------------------------------------------------
# Circle fitting
# ----------------------------------------------------------------------------
def fit_circle(z):
    """Algebraic (Kasa) circle fit. Returns centre (complex), radius, rms err."""
    x, y = z.real, z.imag
    M = np.column_stack([x, y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(M, -(x**2 + y**2), rcond=None)
    c = complex(-sol[0] / 2, -sol[1] / 2)
    R = np.sqrt(max(abs(c) ** 2 - sol[2], 0.0))
    err = np.sqrt(np.mean((np.abs(z - c) - R) ** 2)) if R > 0 else np.inf
    return c, R, err


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def circle_point_estimate(w, z):
    """Classical circle fit for one FRF (receptance points z at rad/s w).

    On a hysteretic-mode circle the angle about the centre is
        phi(w) = phi0 + s * 2*atan(u),  u = (1 - w^2/wr^2) / eta
    Returns (wr, eta, radius, circle error / radius) or None.
    """
    c, R, err = fit_circle(z)
    if not np.isfinite(R) or R == 0:
        return None
    phi = np.unwrap(np.angle(z - c))
    w2 = w**2
    s = 1.0 if phi[-1] < phi[0] else -1.0

    def model(wr, eta):
        return s * 2 * np.arctan((1 - w2 / wr**2) / eta)

    def best_phi0(m):
        return np.angle(np.mean(np.exp(1j * (phi - m))))

    # coarse grid search for a robust start, then least-squares polish
    best = (np.inf, None)
    for wr in np.linspace(w[0], w[-1], 60):
        for eta in np.geomspace(1e-4, MAX_ETA, 50):
            m = model(wr, eta)
            cost = np.sum(_wrap(phi - m - best_phi0(m)) ** 2)
            if cost < best[0]:
                best = (cost, (wr, eta))
    wr0, eta0 = best[1]
    try:
        r = least_squares(lambda p: _wrap(phi - model(p[0], p[1]) - p[2]),
                          [wr0, eta0, best_phi0(model(wr0, eta0))],
                          bounds=([w[0], 1e-5, -np.inf], [w[-1], MAX_ETA, np.inf]))
        wr, eta = r.x[0], r.x[1]
    except Exception:
        wr, eta = wr0, eta0
    return wr, eta, R, err / R


def mode_basis(w, wr, eta):
    return 1.0 / (wr**2 - w**2 + 1j * eta * wr**2)


def fit_modal_constants(w, Z, wr, eta):
    """Per-point A_p (modal constant) and B_p (circle offset) for fixed wr, eta."""
    M = np.column_stack([mode_basis(w, wr, eta), np.ones_like(w, dtype=complex)])
    sol, *_ = np.linalg.lstsq(M, Z.T, rcond=None)
    return sol[0], sol[1], M


def refine_global(w, Z, wr0, eta0):
    def resid(p):
        A, B, M = fit_modal_constants(w, Z, p[0], np.exp(p[1]))
        e = (M @ np.vstack([A, B])).T - Z
        return np.concatenate([e.real.ravel(), e.imag.ravel()])

    r = least_squares(resid, [wr0, np.log(eta0)],
                      bounds=([w[0], np.log(1e-5)], [w[-1], np.log(MAX_ETA)]),
                      x_scale=[w[1] - w[0], 0.5])
    return r.x[0], float(np.exp(r.x[1]))


def weighted_median(x, w):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


def rms_level(R):
    return np.sqrt(np.mean(np.abs(R) ** 2, axis=0))


def mif_db(R):
    return 20 * np.log10(rms_level(R) + 1e-30)


def auto_half_width(level, k, min_side=2, max_side=40):
    """About 2x the half-power half-width of the peak at bin k."""
    half = level[k] / np.sqrt(2)
    lo = k
    while lo > 0 and level[lo - 1] > half and k - lo < max_side:
        lo -= 1
    hi = k
    while hi < len(level) - 1 and level[hi + 1] > half and hi - k < max_side:
        hi += 1
    # don't run into a neighbouring peak: stop at the first local minimum
    left = 1
    while left < max_side and k - left - 1 > 0 and level[k - left - 1] < level[k - left]:
        left += 1
    right = 1
    while right < max_side and k + right + 1 < len(level) and level[k + right + 1] < level[k + right]:
        right += 1
    side = 2 * max(k - lo, hi - k, 1)
    side = min(side, max(left, right))
    return int(np.clip(side, min_side, max_side))


def fit_mode(f, R, k, side, refine=True):
    """Circle-fit the peak at bin k of residual R using bins k-side..k+side."""
    w = 2 * np.pi * f
    idx = np.arange(max(k - side, 1), min(k + side, len(f) - 1) + 1)
    wb = w[idx]
    Zrec = R[:, idx] / (-wb**2)

    ests = []
    for p in range(Zrec.shape[0]):
        e = circle_point_estimate(wb, Zrec[p])
        if e is not None and np.all(np.isfinite(e)):
            ests.append((p,) + tuple(e))
    if not ests:
        return None
    ests = np.array(ests)
    Rmax = ests[:, 3].max()
    good = (ests[:, 3] > 0.2 * Rmax) & (ests[:, 4] < 0.3)
    if not good.any():
        good = ests[:, 3] >= np.sort(ests[:, 3])[-min(5, len(ests))]
    wts = ests[good, 3] ** 2
    wr_c = weighted_median(ests[good, 1], wts)
    eta_c = weighted_median(ests[good, 2], wts)
    wr, eta = refine_global(wb, Zrec, wr_c, eta_c) if refine else (wr_c, eta_c)
    A, B, _ = fit_modal_constants(wb, Zrec, wr, eta)

    contrib_band = (-(wb**2) * mode_basis(wb, wr, eta))[None, :] * A[:, None]
    before = np.sum(np.abs(R[:, idx]) ** 2)
    after = np.sum(np.abs(R[:, idx] - contrib_band) ** 2)
    return dict(peak_hz=f[k], k=k, side=side, idx=idx, fr=wr / 2 / np.pi, eta=eta,
                fr_circle=wr_c / 2 / np.pi, eta_circle=eta_c, A=A, B=B,
                n_circles=int(good.sum()),
                gain_db=10 * np.log10(before / max(after, 1e-300)))


def contribution(m, w):
    wr = 2 * np.pi * m["fr"]
    return (-(w**2) * mode_basis(w, wr, m["eta"]))[None, :] * m["A"][:, None]


def fine_grid(f_lo, f_hi, df, modes, over=8):
    """Frequencies `over` times denser than the measurement, plus extra-dense
    points across each mode's half-power band so every peak is fully drawn."""
    parts = [np.arange(f_lo, f_hi, df / over)]
    for m in modes:
        hw = max(m["eta"] * m["fr"] / 2, df / over)
        parts.append(np.linspace(m["fr"] - 8 * hw, m["fr"] + 8 * hw, 161))
    g = np.concatenate(parts)
    return np.unique(g[(g >= f_lo) & (g <= f_hi) & (g > 0)])


def model_db(modes, f):
    """rms over points of the sum of the given modes, in dB, at any frequencies f.
    Uses sum_p |sum_m A_pm g_m|^2 = g^H (A^H A) g, so no points x f array is built."""
    w = 2 * np.pi * np.asarray(f, float)
    A = np.column_stack([m["A"] for m in modes])                      # points x modes
    G = np.vstack([-(w**2) * mode_basis(w, 2 * np.pi * m["fr"], m["eta"]) for m in modes])
    p = np.einsum("mf,mf->f", G.conj(), (A.conj().T @ A) @ G).real / A.shape[0]
    return 10 * np.log10(np.maximum(p, 0) + 1e-60)


# ----------------------------------------------------------------------------
# Fitting session: residual bookkeeping
# ----------------------------------------------------------------------------
class Session:
    def __init__(self, ds, fmin, fmax, refine):
        self.ds, self.fmin, self.fmax, self.refine = ds, fmin, fmax, refine
        self.f = ds["f"]
        self.w = 2 * np.pi * self.f
        self.R = ds["H"].copy()
        self.modes = []
        self.band = (self.f >= fmin) & (self.f <= fmax)

    def residual_db(self):
        """Residual level re measured, dB, averaged over frequency in [fmin, fmax]."""
        return float(np.mean(mif_db(self.R[:, self.band])
                             - mif_db(self.ds["H"][:, self.band])))

    def snap(self, f_click, search_bins=4):
        k = int(round(f_click / self.ds["df"]))
        lo, hi = max(k - search_bins, 1), min(k + search_bins, len(self.f) - 2)
        lv = rms_level(self.R[:, lo:hi + 1])
        return lo + int(np.argmax(lv))

    def propose(self, f_click, side=None, snap=True):
        k = self.snap(f_click) if snap else int(round(f_click / self.ds["df"]))
        if side is None:
            lo, hi = max(k - 60, 1), min(k + 60, len(self.f) - 1)
            side = auto_half_width(rms_level(self.R[:, lo:hi]), k - lo)
        return fit_mode(self.f, self.R, k, side, self.refine)

    def accept(self, m):
        zz = np.arange(max(m["idx"][0] - max(10, 3 * len(m["idx"])), 1),
                       min(m["idx"][-1] + max(10, 3 * len(m["idx"])), len(self.f) - 1))
        m["zz"] = zz
        m["R_before_zz"] = self.R[:, zz].copy()
        self.R = self.R - contribution(m, self.w)
        m["n"] = len(self.modes) + 1
        m["residual_after"] = self.residual_db()
        self.modes.append(m)

    def undo(self):
        if self.modes:
            m = self.modes.pop()
            self.R = self.R + contribution(m, self.w)
            return m

    def next_peak(self):
        """Lowest residual peak above the last accepted mode (or fmin)."""
        from scipy.signal import find_peaks
        start = max([self.fmin] + [m["fr"] + self.ds["df"] for m in self.modes[-1:]])
        sel = np.where((self.f >= start) & (self.f <= self.fmax))[0]
        if len(sel) < 3:
            return None
        pk, _ = find_peaks(mif_db(self.R[:, sel]), prominence=6)
        return self.f[sel[pk[0]]] if len(pk) else None


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def real_shape(A):
    """Rotate the complex mode shape so it is as real as possible."""
    ang = 0.5 * np.angle(np.sum(A**2))
    s = (A * np.exp(-1j * ang)).real
    return s / np.max(np.abs(s))


def plot_shape(ax, ds, A, title):
    s = real_shape(A)
    if ds["xy"] is not None:
        x, y = ds["xy"][:, 0], ds["xy"][:, 1]
        xs, ys = np.unique(x), np.unique(y)
        if len(xs) * len(ys) == len(s) and len(xs) > 1 and len(ys) > 1:
            grid = np.full((len(ys), len(xs)), np.nan)
            for v, xi, yi in zip(s, x, y):
                grid[np.searchsorted(ys, yi), np.searchsorted(xs, xi)] = v
            dx, dy = (xs[1] - xs[0]) / 2, (ys[1] - ys[0]) / 2
            ax.imshow(grid, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
                      extent=[xs[0] - dx, xs[-1] + dx, ys[0] - dy, ys[-1] + dy],
                      interpolation="bicubic")
            ax.contour(xs, ys, grid, levels=[0], colors="k", linewidths=1)
            ax.plot(x, y, ".", color="k", ms=2)
        else:
            ax.scatter(x, y, c=s, cmap="RdBu_r", vmin=-1, vmax=1, s=60)
        ax.set_aspect("equal")
    else:
        ax.bar(ds["labels"], s)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def plot_nyquist(ax, ds, m, R_band, p, legend=False):
    f = ds["f"]
    w = 2 * np.pi * f
    idx = m["idx"]
    wr = 2 * np.pi * m["fr"]
    wf = np.linspace(w[idx[0]], w[idx[-1]], 400)
    z = R_band[p] / (-w[idx] ** 2)
    model = m["A"][p] * mode_basis(wf, wr, m["eta"]) + m["B"][p]
    full = m["A"][p] * mode_basis(np.linspace(0.5 * wr, 1.5 * wr, 800), wr, m["eta"]) + m["B"][p]
    ax.plot(full.real, full.imag, ":", color="C1", lw=0.8)
    ax.plot(model.real, model.imag, "-", color="C1", lw=1.4, label="fit circle")
    ax.plot(z.real, z.imag, "o-", color="C0", ms=4, lw=0.6, label="data")
    zr = m["A"][p] * mode_basis(np.array([wr]), wr, m["eta"]) + m["B"][p]
    ax.plot(zr.real, zr.imag, "r*", ms=11, label="f_r")
    ax.plot(m["B"][p].real, m["B"][p].imag, "k+", ms=8, label="offset")
    # origin: grey axis lines plus a marker (which also keeps it inside the view)
    ax.axhline(0, color="0.55", lw=0.7, zorder=0)
    ax.axvline(0, color="0.55", lw=0.7, zorder=0)
    ax.plot(0, 0, "o", mfc="none", mec="k", ms=7, mew=1.2, label="origin (0,0)")
    ax.annotate("(0,0)", (0, 0), xytext=(5, 5), textcoords="offset points", fontsize=7)
    ax.set_aspect("equal", "datalim")
    ax.set_title(f"point {ds['labels'][p]}: receptance H/(-w²)", fontsize=9)
    ax.tick_params(labelsize=7)
    if legend:
        ax.legend(fontsize=7, loc="best")


def plot_zoom(ax, ds, m, R_before_zz, zz):
    f = ds["f"]
    w = 2 * np.pi * f
    c = contribution(m, w[zz])
    ff = fine_grid(f[zz[0]], f[zz[-1]], ds["df"], [m], over=10)
    ax.plot(f[zz], mif_db(ds["H"][:, zz]), color="0.7", lw=1, label="measured")
    ax.plot(f[zz], mif_db(R_before_zz), ".-", color="C0", lw=1.3, ms=3,
            label="residual before (bins)")
    ax.plot(ff, model_db([m], ff), color="C1", lw=1.2, label="this mode (smooth)")
    ax.plot(m["fr"], model_db([m], [m["fr"]])[0], "r*", ms=9)
    ax.plot(f[zz], mif_db(R_before_zz - c), color="C2", lw=1.3, label="residual after")
    ax.axvspan(f[m["idx"][0]], f[m["idx"][-1]], color="C1", alpha=0.1)
    ax.axvline(m["fr"], color="r", lw=0.8, ls="--")
    ax.set_xlabel("Hz", fontsize=8)
    ax.set_ylabel("rms |H| (dB)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def plot_model_frf(ax, ds, modes, f_lo, f_hi, proposal=None):
    """Measured rms |H| at its bins next to the fitted modes on a fine grid,
    with every mode's natural frequency marked and labelled."""
    f = ds["f"]
    v = (f >= f_lo) & (f <= f_hi)
    ax.plot(f[v], mif_db(ds["H"][:, v]), ".-", color="0.65", lw=0.6, ms=1.5,
            label=f"measured (a bin every {ds['df']:.3f} Hz)")
    if modes:
        ff = fine_grid(f_lo, f_hi, ds["df"], modes)
        ax.plot(ff, model_db(modes, ff), color="C1", lw=1.0,
                label=f"fitted model, {len(modes)} mode{'s' if len(modes) > 1 else ''} "
                      "(smooth: 8x finer grid)")
        for m in modes:
            y = model_db(modes, [m["fr"]])[0]
            ax.plot(m["fr"], y, "v", color="C3", ms=4)
            ax.annotate(f"{m['fr']:.2f}", (m["fr"], y), xytext=(0, 5),
                        textcoords="offset points", rotation=90, fontsize=7,
                        ha="center", va="bottom", color="C3")
    if proposal is not None:
        ff = fine_grid(f_lo, f_hi, ds["df"], [proposal])
        ff = ff[np.abs(ff - proposal["fr"]) < max(40 * proposal["eta"] * proposal["fr"],
                                                   20 * ds["df"])]
        ax.plot(ff, model_db([proposal], ff), "--", color="C2", lw=1.2,
                label=f"proposed mode, {proposal['fr']:.2f} Hz")
    ax.set_xscale("log")
    ax.set_xlabel("Hz")
    ax.set_ylabel("rms |H| over points (dB)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=8)


def mode_title(m):
    return (f"f_r = {m['fr']:.3f} Hz   eta = {m['eta']:.5f}  (Q = {1/m['eta']:.0f})   "
            f"[circle-only {m['fr_circle']:.3f} Hz, eta {m['eta_circle']:.5f}]   "
            f"band {len(m['idx'])} bins   local improvement {m['gain_db']:.1f} dB")


# ----------------------------------------------------------------------------
# Interactive GUI
# ----------------------------------------------------------------------------
class FitGUI:
    HELP = ("click a peak = fit it   n = next peak   a/Enter = accept   x/Esc = discard   "
            "[ ] = band -/+   u = undo   \u2190 \u2192 = browse steps   f = model FRF window   "
            "s = save   q = save & quit")

    def __init__(self, sess, outdir):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        self.plt = plt
        self.s, self.outdir = sess, outdir
        self.proposal = None
        self.step = None            # None = live (latest); else number of modes shown
        self.fig = plt.figure(figsize=(16, 9.5))
        gs = self.fig.add_gridspec(2, 4, height_ratios=[1.15, 1])
        self.ax_top = self.fig.add_subplot(gs[0, :])
        self.ax_ny = [self.fig.add_subplot(gs[1, 0]), self.fig.add_subplot(gs[1, 1])]
        self.ax_zoom = self.fig.add_subplot(gs[1, 2])
        self.ax_shape = self.fig.add_subplot(gs[1, 3])
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        for k in ("keymap.save", "keymap.quit", "keymap.back", "keymap.forward"):
            plt.rcParams[k] = []
        plt.rcParams["keymap.fullscreen"] = ["ctrl+f"]    # free up f
        self.set_title(self.fig, "circle fit")
        f = sess.f
        self.view = (f >= sess.fmin * 0.8) & (f <= sess.fmax)
        self.ax_top.set_xscale("log")
        self.draw_top(keep_limits=False)
        self.clear_bottom("Click a peak in the top panel (or press n) to fit the next mode.")
        self.fig.tight_layout(rect=[0, 0, 1, 0.93])

        # step-browser buttons (top right)
        self.btn_prev = Button(self.fig.add_axes([0.80, 0.945, 0.055, 0.035]), "\u25c0 Prev")
        self.btn_next = Button(self.fig.add_axes([0.86, 0.945, 0.055, 0.035]), "Next \u25b6")
        self.btn_live = Button(self.fig.add_axes([0.92, 0.945, 0.06, 0.035]), "Latest")
        self.btn_prev.on_clicked(lambda e: self.browse(-1))
        self.btn_next.on_clicked(lambda e: self.browse(+1))
        self.btn_live.on_clicked(lambda e: self.go_live())

        self.fig2 = None
        self.open_model_window()

    # -- second window: fitted model on a fine frequency grid -----------------
    def set_title(self, fig, what):
        try:
            fig.canvas.manager.set_window_title(f"{self.s.ds['name']}: {what}")
        except Exception:
            pass

    def open_model_window(self):
        self.fig2 = self.plt.figure(figsize=(15, 6.5))
        self.ax_model = self.fig2.add_subplot(111)
        self.set_title(self.fig2, "fitted model FRF")
        self.fig2.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig2.canvas.mpl_connect("close_event", self.on_model_closed)
        self.draw_model(keep_limits=False)
        self.fig2.tight_layout()

    def on_model_closed(self, ev):
        self.fig2 = None

    def draw_model(self, keep_limits=True):
        if self.fig2 is None:
            return
        ax, s = self.ax_model, self.s
        lims = (ax.get_xlim(), ax.get_ylim()) if keep_limits else None
        ax.clear()
        n = self.shown()
        plot_model_frf(ax, s.ds, s.modes[:n], s.fmin * 0.8, s.fmax,
                       None if self.browsing() else self.proposal)
        what = f"step {n} of {len(s.modes)}" if self.browsing() else f"{n} modes accepted"
        ax.set_title(f"{s.ds['name']}: fitted model ({what}). Labels are the fitted natural "
                     "frequencies in Hz; use the toolbar zoom to look closely.", fontsize=10)
        if lims:
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])
        self.fig2.canvas.draw_idle()

    # -- state ----------------------------------------------------------
    def shown(self):
        """Number of modes in the step being displayed."""
        return len(self.s.modes) if self.step is None else self.step

    def browsing(self):
        return self.step is not None

    def residual_shown(self, sel):
        """Residual after the first shown() modes, on frequency selection sel."""
        if not self.browsing():
            return self.s.R[:, sel]
        R = self.s.ds["H"][:, sel].copy()
        w = self.s.w[sel]
        for m in self.s.modes[:self.step]:
            R -= contribution(m, w)
        return R

    # -- drawing --------------------------------------------------------
    def draw_top(self, keep_limits=True):
        ax, s, ds = self.ax_top, self.s, self.s.ds
        lims = (ax.get_xlim(), ax.get_ylim()) if keep_limits else None
        ax.clear()
        n = self.shown()
        f = s.f[self.view]
        Rv = self.residual_shown(self.view)
        ax.plot(f, mif_db(ds["H"][:, self.view]), color="0.6", lw=0.8, label="measured")
        if n:
            ax.plot(f, mif_db(ds["H"][:, self.view] - Rv), color="C1", lw=0.8,
                    label=f"sum of {n} mode{'s' if n > 1 else ''}")
        ax.plot(f, mif_db(Rv), color="C0", lw=0.8, label=f"residual after {n} modes")
        for m in s.modes[:n]:
            ax.axvline(m["fr"], color="C1", lw=0.5, alpha=0.6)
        for m in s.modes[n:]:        # later modes, when browsing back
            ax.axvline(m["fr"], color="0.5", lw=0.5, ls=":", alpha=0.8)
        if self.browsing() and n:
            ax.axvline(s.modes[n - 1]["fr"], color="r", ls="--", lw=0.9)
        if self.proposal:
            p = self.proposal
            zz = np.arange(max(p["idx"][0] - 20, 1), min(p["idx"][-1] + 20, len(s.f) - 1))
            ax.plot(s.f[zz], mif_db(s.R[:, zz] - contribution(p, s.w[zz])), color="C2",
                    lw=1.2, label="residual after proposed mode")
            ax.axvline(p["fr"], color="r", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_ylabel("rms |H| over points (dB)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper left", fontsize=8)
        sel = s.band & self.view
        res = float(np.mean(mif_db(self.residual_shown(sel)) - mif_db(ds["H"][:, sel])))
        if self.browsing():
            head = (f"VIEWING STEP {n} of {len(s.modes)}"
                    + (f" (after mode {n}, {s.modes[n-1]['fr']:.2f} Hz)" if n else " (no modes)")
                    + "  \u2014  \u2192 / Latest to return")
            color = "C3"
        else:
            head = f"{len(s.modes)} modes accepted"
            color = "k"
        ax.set_title(f"{ds['name']}:  {head},  residual {res:.2f} dB re measured "
                     f"(mean over {s.fmin:g}\u2013{s.fmax:g} Hz)", fontsize=10, color=color)
        if lims:
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])
        self.fig.suptitle(self.HELP, fontsize=9, color="0.3", x=0.4)

    def clear_bottom(self, msg):
        for a in self.ax_ny + [self.ax_zoom, self.ax_shape]:
            a.clear()
            a.set_xticks([])
            a.set_yticks([])
        self.ax_zoom.text(0.5, 0.5, msg, ha="center", va="center", wrap=True,
                          transform=self.ax_zoom.transAxes, fontsize=9)

    def draw_fit(self, m, R_band, R_zz, zz, title, color):
        for a in self.ax_ny + [self.ax_zoom, self.ax_shape]:
            a.clear()
        order = np.argsort(-np.abs(m["A"]))[:2]
        for j, (a, pt) in enumerate(zip(self.ax_ny, order)):
            plot_nyquist(a, self.s.ds, m, R_band, pt, legend=(j == 0))
        plot_zoom(self.ax_zoom, self.s.ds, m, R_zz, zz)
        plot_shape(self.ax_shape, self.s.ds, m["A"], "mode shape (real part)")
        self.ax_zoom.set_title(f"{title}: {m['fr']:.3f} Hz, eta {m['eta']:.5f}, "
                               f"{len(m['idx'])} bins, {m['gain_db']:.1f} dB better",
                               fontsize=8, color=color)

    def draw_proposal(self):
        p, s = self.proposal, self.s
        pad = max(10, 3 * len(p["idx"]))
        zz = np.arange(max(p["idx"][0] - pad, 1), min(p["idx"][-1] + pad, len(s.f) - 1))
        self.draw_fit(p, s.R[:, p["idx"]], s.R[:, zz], zz, "PROPOSED", "C3")

    def draw_accepted(self, n):
        """Bottom panels for accepted mode n (1-based), as it was fitted."""
        m = self.s.modes[n - 1]
        off = m["idx"][0] - m["zz"][0]
        R_band = m["R_before_zz"][:, off:off + len(m["idx"])]
        self.draw_fit(m, R_band, m["R_before_zz"], m["zz"], f"MODE {n}", "C0")

    def redraw(self):
        self.draw_top()
        self.fig.canvas.draw_idle()
        self.draw_model()

    # -- step browsing ----------------------------------------------------
    def browse(self, d):
        N = len(self.s.modes)
        n = min(max(self.shown() + d, 0), N)
        if n == self.shown() and self.browsing() == (n != N):
            return
        self.proposal = None
        self.step = None if n == N else n
        if n:
            self.draw_accepted(n)
        else:
            self.clear_bottom("Step 0: measured data, no modes removed.")
        self.redraw()

    def go_live(self):
        if self.browsing():
            self.browse(len(self.s.modes) - self.shown())

    def leave_browsing(self):
        if self.browsing():
            print("  (returning to the latest step to fit)")
            self.step = None

    # -- actions ----------------------------------------------------------
    def fit_at(self, fhz, side=None, snap=True):
        self.leave_browsing()
        p = self.s.propose(fhz, side, snap)
        if p is None:
            print(f"  could not fit a circle near {fhz:.2f} Hz")
            return
        self.proposal = p
        print(f"  proposed: {mode_title(p)}")
        self.draw_proposal()
        self.redraw()

    def on_click(self, ev):
        tb = getattr(self.fig.canvas, "toolbar", None)
        if tb is not None and str(getattr(tb, "mode", "")):
            return                      # zoom / pan active
        if ev.inaxes is self.ax_top and ev.xdata and ev.button == 1:
            self.fit_at(ev.xdata)

    def on_key(self, ev):
        k = ev.key
        if k == "left":
            self.browse(-1)
        elif k == "right":
            self.browse(+1)
        elif k in ("a", "enter") and self.proposal:
            self.s.accept(self.proposal)
            m = self.s.modes[-1]
            print(f"  ACCEPTED mode {m['n']}: {m['fr']:.3f} Hz, eta {m['eta']:.5f}; "
                  f"residual {m['residual_after']:.2f} dB")
            self.proposal = None
            self.draw_accepted(m["n"])
            self.ax_zoom.set_title(f"ACCEPTED mode {m['n']} at {m['fr']:.2f} Hz \u2014 "
                                   "click the next peak or press n", fontsize=8, color="C2")
            self.redraw()
        elif k in ("x", "escape"):
            self.proposal = None
            self.clear_bottom("Proposal discarded. Click a peak or press n.")
            self.redraw()
        elif k in ("[", "]") and self.proposal:
            side = max(2, self.proposal["side"] + (1 if k == "]" else -1))
            self.fit_at(self.proposal["peak_hz"], side=side, snap=False)
        elif k == "n":
            self.leave_browsing()
            fn = self.s.next_peak()
            if fn is None:
                print("  no further peaks above the last mode")
                self.redraw()
            else:
                self.fit_at(fn)
        elif k == "u":
            self.step = None
            m = self.s.undo()
            if m:
                print(f"  undid mode {m['n']} ({m['fr']:.2f} Hz)")
                self.proposal = None
                self.clear_bottom(f"Undid mode {m['n']} ({m['fr']:.2f} Hz).")
            self.redraw()
        elif k == "f":
            if self.fig2 is None:
                self.open_model_window()
                self.fig2.show()
            else:
                self.draw_model()
        elif k == "s":
            save_results(self.s, self.outdir)
        elif k == "q":
            save_results(self.s, self.outdir)
            if self.fig2 is not None:
                self.plt.close(self.fig2)
            self.plt.close(self.fig)

    def run(self):
        self.plt.show()


# ----------------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------------
def save_results(sess, outdir):
    import matplotlib.pyplot as plt
    ds, modes = sess.ds, sess.modes
    os.makedirs(os.path.join(outdir, "steps"), exist_ok=True)
    for old in glob.glob(os.path.join(outdir, "steps", "mode_*.png")):
        os.remove(old)

    with open(os.path.join(outdir, "modes.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["mode", "f_r_Hz", "eta", "zeta", "Q", "f_r_circle_only_Hz",
                     "eta_circle_only", "band_lo_Hz", "band_hi_Hz", "peak_Hz",
                     "half_width_bins", "n_points_used_for_circle_estimate",
                     "local_improvement_dB", "residual_after_dB_re_measured"])
        for m in modes:
            wr.writerow([m["n"], f"{m['fr']:.4f}", f"{m['eta']:.6f}", f"{m['eta']/2:.6f}",
                         f"{1/m['eta']:.1f}", f"{m['fr_circle']:.4f}",
                         f"{m['eta_circle']:.6f}", f"{ds['f'][m['idx'][0]]:.2f}",
                         f"{ds['f'][m['idx'][-1]]:.2f}", f"{m['peak_hz']:.4f}",
                         m["side"], m["n_circles"], f"{m['gain_db']:.2f}",
                         f"{m['residual_after']:.3f}"])
    with open(os.path.join(outdir, "mode_shapes.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        hdr = ["point", "x_mm", "y_mm"]
        for m in modes:
            hdr += [f"m{m['n']}_{m['fr']:.1f}Hz_real", f"m{m['n']}_absA",
                    f"m{m['n']}_phaseA_deg"]
        wr.writerow(hdr)
        shapes = [real_shape(m["A"]) for m in modes]
        for p, lab in enumerate(ds["labels"]):
            row = [lab] + (list(ds["xy"][p]) if ds["xy"] is not None else ["", ""])
            for m, s in zip(modes, shapes):
                row += [f"{s[p]:.5f}", f"{abs(m['A'][p]):.6g}",
                        f"{np.degrees(np.angle(m['A'][p])):.2f}"]
            wr.writerow(row)

    # one figure per accepted step
    for m in modes:
        fig = plt.figure(figsize=(15, 7.5))
        gs = fig.add_gridspec(2, 3)
        order = np.argsort(-np.abs(m["A"]))[:3]
        off = m["idx"][0] - m["zz"][0]
        Rb_band = m["R_before_zz"][:, off:off + len(m["idx"])]
        for j, p in enumerate(order):
            plot_nyquist(fig.add_subplot(gs[0, j]), ds, m, Rb_band, p, legend=(j == 0))
        plot_zoom(fig.add_subplot(gs[1, :2]), ds, m, m["R_before_zz"], m["zz"])
        plot_shape(fig.add_subplot(gs[1, 2]), ds, m["A"], "mode shape (real part)")
        fig.suptitle(f"{ds['name']} mode {m['n']}:  " + mode_title(m), fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "steps", f"mode_{m['n']:02d}_{m['fr']:.1f}Hz.png"),
                    dpi=100)
        plt.close(fig)

    # summary
    f = ds["f"]
    v = (f >= sess.fmin * 0.8) & (f <= sess.fmax)
    fig, axs = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw=dict(height_ratios=[3, 1.3]))
    ax = axs[0]
    ax.semilogx(f[v], mif_db(ds["H"][:, v]), color="k", lw=0.8, label="measured")
    if modes:
        ax.semilogx(f[v], mif_db(ds["H"][:, v] - sess.R[:, v]), color="C1", lw=0.8,
                    label=f"sum of {len(modes)} modes")
    ax.semilogx(f[v], mif_db(sess.R[:, v]), color="C2", lw=0.6, label="residual")
    for m in modes:
        ax.axvline(m["fr"], color="r", lw=0.4, alpha=0.4)
    ax.set_ylabel("rms |H| over points (dB)")
    ax.set_title(f"{ds['name']}: incremental circle fit")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    hist = [0.0] + [m["residual_after"] for m in modes]
    axs[1].plot(range(len(hist)), hist, "o-", ms=3)
    axs[1].set_xlabel("modes removed")
    axs[1].set_ylabel("mean residual re measured (dB)")
    axs[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "summary.png"), dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 6.5))
    plot_model_frf(ax, ds, modes, sess.fmin * 0.8, sess.fmax)
    ax.set_title(f"{ds['name']}: fitted model on a fine frequency grid "
                 "(labels = natural frequencies, Hz)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "model_frf.png"), dpi=120)
    plt.close(fig)

    if modes:
        cols = min(8, len(modes))
        rows = int(np.ceil(len(modes) / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.6 * rows), squeeze=False)
        for a in axs.ravel():
            a.axis("off")
        for m, a in zip(modes, axs.ravel()):
            a.axis("on")
            plot_shape(a, ds, m["A"], f"{m['n']}: {m['fr']:.1f} Hz")
            a.set_xticks([])
            a.set_yticks([])
        fig.suptitle(f"{ds['name']} mode shapes")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "mode_shapes.png"), dpi=100)
        plt.close(fig)
    print(f"  saved {len(modes)} modes to {outdir}")


# ----------------------------------------------------------------------------
def parse_modes(text):
    out = []
    for tok in filter(None, (t.strip() for t in text.split(","))):
        fhz, _, side = tok.partition(":")
        out.append((float(fhz), int(side) if side else None))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="ObieApp test folder (contains TRF/), e.g. Demo/Demo_07")
    ap.add_argument("--fmin", type=float, default=20.0, help="plot/residual range low (Hz)")
    ap.add_argument("--fmax", type=float, default=7000.0, help="plot/residual range high (Hz)")
    ap.add_argument("--modes", default="", help="peaks to pre-fit, e.g. 34.3,73.6,196:4")
    ap.add_argument("--no-gui", action="store_true", help="fit --modes, save, exit")
    ap.add_argument("--no-refine", action="store_true",
                    help="classical per-point circle estimates only (no global refinement)")
    ap.add_argument("--out", default=None,
                    help="output folder (default: CircleFit/<test name> next to this script)")
    args = ap.parse_args()

    if args.no_gui:
        matplotlib.use("Agg")
    ds = load_dataset(args.folder)
    outdir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "CircleFit", ds["name"])
    print(f"{ds['name']}: {len(ds['labels'])} points, df = {ds['df']:.4f} Hz")
    sess = Session(ds, args.fmin, args.fmax, not args.no_refine)

    for fhz, side in parse_modes(args.modes):
        m = sess.propose(fhz, side)
        if m is None:
            print(f"  could not fit near {fhz} Hz")
            continue
        sess.accept(m)
        print(f"  mode {m['n']}: {mode_title(m)}")

    if args.no_gui:
        save_results(sess, outdir)
    else:
        FitGUI(sess, outdir).run()


if __name__ == "__main__":
    main()

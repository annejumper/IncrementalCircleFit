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
import functools
import glob
import json
import os
import re
import itertools
import struct
import sys

import numpy as np
from scipy.optimize import least_squares

import matplotlib

HEADER_BYTES = 110
_FIT_IDS = itertools.count(1)   # unique id per fit, so saved zoom never leaks between fits
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
    side = int(np.clip(side, min_side, max_side))
    # never let the band reach a point higher than this peak (a bigger neighbour)
    while side > min_side and np.max(level[max(k - side, 0):k + side + 1]) > level[k]:
        side -= 1
    return side


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
    # per-point circle estimates, for the estimates panel. status: 0 used,
    # 1 excluded (circle < 20% of largest), 2 excluded (data > 30% of R off its
    # circle), 3 no circle could be fitted
    P = Zrec.shape[0]
    est = dict(fr=np.full(P, np.nan), eta=np.full(P, np.nan), R=np.zeros(P),
               relerr=np.full(P, np.nan), status=np.full(P, 3))
    pi = ests[:, 0].astype(int)
    est["fr"][pi] = ests[:, 1] / 2 / np.pi
    est["eta"][pi] = ests[:, 2]
    est["R"][pi] = ests[:, 3]
    est["relerr"][pi] = ests[:, 4]
    est["status"][pi] = np.where(good, 0, np.where(ests[:, 3] <= 0.2 * Rmax, 1, 2))
    return dict(uid=next(_FIT_IDS), peak_hz=f[k], k=k, side=side, idx=idx,
                fr=wr / 2 / np.pi, eta=eta,
                fr_circle=wr_c / 2 / np.pi, eta_circle=eta_c, A=A, B=B, est=est,
                n_circles=int(good.sum()),
                gain_db=10 * np.log10(before / max(after, 1e-300)))


def contribution(m, w):
    wr = 2 * np.pi * m["fr"]
    return (-(w**2) * mode_basis(w, wr, m["eta"]))[None, :] * m["A"][:, None]


@functools.lru_cache(maxsize=None)
def _ritz_1d(n_terms, kind):
    """1-D Legendre trial functions on [0, 1] and their Gauss-quadrature inner
    products. kind: 'clamped' x^2 P_i (w = w' = 0 at x = 0), 'pinned' x P_i
    (w = 0 at x = 0, slope free), 'free' P_i. Returns int X X, int X'X',
    int X''X'', int X''_i X_k and X'_i(0)."""
    from numpy.polynomial import legendre as L
    xq, wq = L.leggauss(2 * n_terms + 10)
    xq, wq = (xq + 1) / 2, wq / 2
    s = 2 * xq - 1
    B0, B1, B2, g0 = [], [], [], []
    for i in range(n_terms):
        c = np.zeros(i + 1)
        c[-1] = 1
        P = L.legval(s, c)
        dP = 2 * L.legval(s, L.legder(c)) if i >= 1 else 0 * s
        d2P = 4 * L.legval(s, L.legder(c, 2)) if i >= 2 else 0 * s
        if kind == "clamped":
            B0.append(xq**2 * P)
            B1.append(2 * xq * P + xq**2 * dP)
            B2.append(2 * P + 4 * xq * dP + xq**2 * d2P)
            g0.append(0.0)
        elif kind == "pinned":
            B0.append(xq * P)
            B1.append(P + xq * dP)
            B2.append(2 * dP + xq * d2P)
            g0.append((-1.0) ** i)                     # P_i(-1)
        else:
            B0.append(P)
            B1.append(dP)
            B2.append(d2P)
            g0.append(np.nan)
    B0, B1, B2 = map(np.array, (B0, B1, B2))
    ip = lambda U, V: (U * wq) @ V.T
    return ip(B0, B0), ip(B1, B1), ip(B2, B2), ip(B2, B0), np.array(g0)


def plate_frequencies(length_mm, width_mm, thick_mm, clamp_mm, E, nu, rho, fmax,
                      n_terms=24, k_rot=None, n_max=None):
    """Natural frequencies (Hz) of a thin rectangular cantilever plate (Kirchhoff
    theory), clamped along one full edge over a strip clamp_mm deep and free on
    the other three edges. The free span is length_mm - clamp_mm, perpendicular
    to the clamped edge; width_mm runs along it.

    k_rot: None for a rigid clamp. Otherwise the clamp holds the edge in place
    but lets it rotate against a rotational spring of k_rot N*m/rad per metre
    of edge (an elastic clamp; very large k_rot tends to the rigid clamp).

    Rayleigh-Ritz with separable Legendre-polynomial trial functions across the
    span and P_j(2y-1) along the width. Converges from above; with 24 x 24 terms
    the first ~50 modes are well converged.
    """
    from scipy.linalg import eigh
    a = (length_mm - clamp_mm) / 1000.0
    b = width_mm / 1000.0
    h = thick_mm / 1000.0
    D = E * h**3 / (12 * (1 - nu**2))
    X0, X1, X2, X20, g = _ritz_1d(n_terms, "clamped" if k_rot is None else "pinned")
    Y0, Y1, Y2, Y20, _ = _ritz_1d(n_terms, "free")
    K = (np.kron(X2, Y0) / a**4 + np.kron(X0, Y2) / b**4
         + nu * (np.kron(X20, Y20.T) + np.kron(X20.T, Y20)) / (a * b) ** 2
         + 2 * (1 - nu) * np.kron(X1, Y1) / (a * b) ** 2) * D * a * b
    if k_rot is not None:
        # spring energy 1/2 int k_rot (dw/dx at the clamp)^2 dy
        K = K + k_rot * b / a**2 * np.kron(np.outer(g, g), Y0)
    M = np.kron(X0, Y0) * rho * h * a * b
    sub = None if n_max is None else [0, min(n_max, M.shape[0]) - 1]
    ev = eigh(K, M, eigvals_only=True, subset_by_index=sub)
    f = np.sqrt(np.maximum(ev, 0)) / (2 * np.pi)
    return f[f <= fmax]


DEFAULT_PLATE = dict(L=150.0, W=150.0, t=0.86, c=15.0, E=69e9, nu=0.33, rho=2700.0, k=None)


def plate_theory(pl, fmax, n_terms=24, n_max=None):
    return plate_frequencies(pl["L"], pl["W"], pl["t"], pl["c"], pl["E"], pl["nu"], pl["rho"],
                             fmax, n_terms=n_terms, k_rot=pl["k"], n_max=n_max)


def plate_label(pl):
    k = "rigid clamp" if pl["k"] is None else f"clamp k {pl['k']:.3g} N\u00b7m/rad/m"
    return (f"analytical: {pl['L']:g}x{pl['W']:g}x{pl['t']:g} mm, {pl['c']:g} mm clamped "
            f"(span {pl['L'] - pl['c']:g}), E {pl['E'] / 1e9:g} GPa, \u03bd {pl['nu']:g}, "
            f"\u03c1 {pl['rho']:g}, {k}")


def plate_cli(pl):
    return (f"--plate {pl['L']:g},{pl['W']:g},{pl['t']:g},{pl['c']:g} --E {pl['E']:.4g} "
            f"--nu {pl['nu']:g} --rho {pl['rho']:g} --clamp-k "
            + ("rigid" if pl["k"] is None else f"{pl['k']:.3g}"))


def match_modes(measured, theory):
    """Pair each measured frequency with an analytical mode, one-to-one, keeping
    the total log-frequency distance as small as possible. Returns
    (measured Hz, analytical index, analytical Hz, % difference) per measured
    peak, sorted by measured frequency; peaks left over (more peaks than
    analytical modes) are omitted."""
    from scipy.optimize import linear_sum_assignment
    if theory is None or not len(theory) or not len(measured):
        return []
    fm = np.asarray(sorted(measured), float)
    th = np.asarray(theory, float)
    rows, cols = linear_sum_assignment(np.abs(np.log(th[None, :] / fm[:, None])))
    return [(fm[r], int(c), th[c], 100 * (th[c] - fm[r]) / fm[r]) for r, c in zip(rows, cols)]


def optimize_plate(pl, measured, specs, free, fmax, n_terms=14):
    """Adjust the parameters named in `free` (keys of `specs`, searched in slider
    units within the slider ranges) so the analytical frequencies best match
    `measured` (one-to-one pairing, rms of log differences). Returns the new
    parameter dict and the rms % difference before and after."""
    from scipy.optimize import minimize
    spec = {sp[0]: sp for sp in specs}
    fm = np.asarray(sorted(measured), float)
    n_max = len(fm) + 20

    def unpack(u):
        q = dict(pl)
        for key, ui in zip(free, np.clip(u, 0, 1)):
            _, _, lo, hi, _, from_s = spec[key]
            q[key] = from_s(lo + ui * (hi - lo))
        return q

    def cost(q):
        th = plate_theory(q, fmax * 1.5, n_terms=n_terms, n_max=n_max)
        pairs = match_modes(fm, th)
        miss = len(fm) - len(pairs)                  # peaks with no mode to pair with
        d = [np.log(p[2] / p[0]) for p in pairs] + [0.5] * miss
        return float(np.sqrt(np.mean(np.square(d))))

    def obj(u):
        return cost(unpack(u)) + 10 * np.sum(np.clip(u - 1, 0, None) + np.clip(-u, 0, None))

    u0 = []
    for key in free:
        _, _, lo, hi, to_s, _ = spec[key]
        u0.append((to_s(pl[key]) - lo) / (hi - lo))
    u0 = np.clip(np.array(u0, float), 0, 1)
    before = cost(pl)
    # several starts: the current values plus a coarse spread over each free
    # parameter's range, since the peak pairing makes the cost bumpy
    starts = [u0]
    for i in range(len(free)):
        for v in (0.1, 0.35, 0.65, 0.9):
            u = u0.copy()
            u[i] = v
            starts.append(u)
    starts.sort(key=obj)
    best = None
    for u in starts[:4]:
        r = minimize(obj, u, method="Nelder-Mead",
                     options=dict(xatol=1e-3, fatol=1e-5, maxiter=300 * len(free)))
        if best is None or r.fun < best.fun:
            best = r
    q = unpack(best.x)
    return q, 100 * before, 100 * cost(q)


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
        """Nearest local maximum of the residual to the clicked bin (not the
        largest value nearby, so a small peak next to a big one can be picked)."""
        k = int(round(f_click / self.ds["df"]))
        lo, hi = max(k - search_bins - 1, 0), min(k + search_bins + 1, len(self.f) - 1)
        lv = rms_level(self.R[:, lo:hi + 1])
        c = k - lo
        for d in range(search_bins + 1):
            for j in (c - d, c + d):
                if 1 <= j < len(lv) - 1 and lv[j] >= lv[j - 1] and lv[j] >= lv[j + 1]:
                    return lo + j
        return lo + 1 + int(np.argmax(lv[1:-1]))

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


MARK_STYLE = {"strongest": ("C2", "strongest"),
              "poor_fit": ("C3", "poor circle fit"),
              "phase": ("C4", "not cleanly in/out of phase")}


def point_diagnostics(m, R_band, f_band):
    """Per-point checks for one fitted mode, and the points worth a closer look.

    strength  |A_p| relative to the largest
    err       rms distance of the band data from the fitted circle / circle diameter
    dev       how far A_p's phase is from 0 or 180 deg relative to the mode as a whole
              (a clean real mode has every point exactly in or out of phase)
    """
    w = 2 * np.pi * np.asarray(f_band)
    wr = 2 * np.pi * m["fr"]
    A, B = m["A"], m["B"]
    Z = R_band / (-w**2)
    model = A[:, None] * mode_basis(w, wr, m["eta"])[None, :] + B[:, None]
    diam = np.abs(A) / (m["eta"] * wr**2)
    strength = np.abs(A) / np.max(np.abs(A))
    err = np.sqrt(np.mean(np.abs(Z - model) ** 2, axis=1)) / np.maximum(diam, 1e-300)
    ang = 0.5 * np.angle(np.sum(A**2))
    ph = np.degrees(np.angle(A * np.exp(-1j * ang)))
    dev = np.abs((ph + 90) % 180 - 90)
    strong = np.where(strength >= 0.25)[0]       # weak points are noisy by nature

    def worst(x, thresh):
        c = strong[x[strong] > thresh]
        return [int(p) for p in c[np.argsort(-x[c])][:2]]

    picks = {"strongest": [int(p) for p in np.argsort(-strength)[:2]],
             "poor_fit": worst(err, 0.10),
             "phase": worst(dev, 20.0)}
    return dict(strength=strength, err=err, dev=dev, picks=picks)


def describe_point(ds, diag, p):
    return (f"|A| {100 * diag['strength'][p]:.0f}% of max, "
            f"off-circle {100 * diag['err'][p]:.0f}%, phase {diag['dev'][p]:.0f}°")


def plot_shape(ax, ds, A, title, diag=None, selected=None):
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
    if diag is not None or selected is not None:
        pos = (ds["xy"] if ds["xy"] is not None
               else np.column_stack([ds["labels"], s]).astype(float))
        if diag is not None:
            lines = []
            for kind, sz in (("strongest", 9), ("poor_fit", 13), ("phase", 17)):
                col, name = MARK_STYLE[kind]
                pts = diag["picks"][kind]
                for p in pts:
                    ax.plot(*pos[p], "o", mfc="none", mec=col, ms=sz, mew=1.6)
                    ax.annotate(str(ds["labels"][p]), pos[p], xytext=(5, 5),
                                textcoords="offset points", fontsize=7, color=col,
                                fontweight="bold")
                lines.append(f"{name}: " + (", ".join(str(ds["labels"][p]) for p in pts)
                                            if pts else "none"))
            ax.set_xlabel("\n".join(lines), fontsize=7, color="0.2")
        if selected is not None:
            ax.plot(*pos[selected], "s", mfc="none", mec="k", ms=11, mew=1.8)
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


THEORY_COLOR = "C4"


def plot_theory(ax, ds, f_lo, f_hi, numbers=True, legend=True):
    """Analytical plate frequencies as dashed lines, numbered along the top."""
    th = ds.get("theory")
    if th is None:
        return
    import matplotlib.transforms as mtransforms
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    first, last_num = True, None
    texts = []
    for i, ft in enumerate(th):
        if not f_lo <= ft <= f_hi:
            continue
        ax.axvline(ft, color=THEORY_COLOR, ls="--", lw=0.9, alpha=0.75,
                   label=(ds["theory_label"] if first and legend else None))
        first = False
        if numbers and (last_num is None or np.log10(ft / last_num) > 0.012):
            last_num = ft      # skip numbers that would overlap the previous one
            texts.append(ax.text(ft, 1.003, str(i + 1), transform=tr, fontsize=6,
                                 color=THEORY_COLOR, ha="center", va="bottom"))
    # numbers sit above the plot, so clipping can't hide them: hide the ones
    # outside the current x range whenever the view changes
    ax._theory_texts = texts
    def _update(a):
        lo, hi = sorted(a.get_xlim())
        for t in getattr(a, "_theory_texts", []):
            t.set_visible(lo <= t.get_position()[0] <= hi)

    if getattr(ax, "_theory_cb", None) is not ax.callbacks:   # clearing an axes resets these
        ax.callbacks.connect("xlim_changed", _update)
        ax._theory_cb = ax.callbacks


EST_STYLE = {0: dict(marker="o", color="C0", label="used"),
             1: dict(marker="o", color="0.6", mfc="none", label="excluded: circle < 20% of largest"),
             2: dict(marker="x", color="C3", label="excluded: data > 30% of R off its circle"),
             3: dict(marker="|", color="0.4", label="excluded: no circle")}


def plot_estimates(ax, ds, m, selected=None):
    """Each point's own circle-fit f_r, showing which points went into the
    circle-only estimate (weighted median) and which were excluded, against
    the circle-only value and the final value from the global fit of all points."""
    est = m.get("est")
    if est is None:
        return
    x = np.array(ds["labels"], float)
    used = est["status"] == 0
    y = est["fr"] - m["fr"]
    dev = np.abs(y[used])
    lim = max(1.4 * (dev.max() if dev.size else 0), 2 * ds["df"])
    lo, hi = -lim, lim
    for st, sty in EST_STYLE.items():
        sel = est["status"] == st
        if not sel.any():
            continue
        yy = y[sel] if st != 3 else np.zeros(sel.sum())
        yy = np.clip(np.nan_to_num(yy, nan=0.0), lo, hi)      # off-scale points sit on the edge
        sz = (6 + 60 * (est["R"][sel] / max(est["R"].max(), 1e-300)) ** 2) if st == 0 else 18
        if sty["marker"] in ("x", "|"):
            ax.scatter(x[sel], yy, s=sz, marker=sty["marker"], c=sty["color"], linewidths=1,
                       label=sty["label"] + f" ({sel.sum()})", zorder=3)
        else:
            ax.scatter(x[sel], yy, s=sz, marker="o", facecolors=sty.get("mfc", sty["color"]),
                       edgecolors=sty["color"], linewidths=1,
                       label=sty["label"] + f" ({sel.sum()})", zorder=3)
    ax.axhline(0, color="C1", lw=1.2, label=f"final {m['fr']:.3f} Hz (global fit, all points)")
    ax.axhline(m["fr_circle"] - m["fr"], color="C1", lw=1, ls="--",
               label=f"circle-only {m['fr_circle']:.3f} Hz (weighted median of used)")
    if selected is not None:
        yv = 0.0 if est["status"][selected] == 3 else np.clip(y[selected], lo, hi)
        ax.plot(x[selected], yv, "s", mfc="none", mec="k", ms=9, mew=1.6, zorder=4)
    ax.set_ylim(lo, hi)
    ax.set_ylabel("point's f_r estimate \u2212 final (Hz)", fontsize=7)
    ax.set_xlabel("point (marker size = circle size = weight)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    ax.set_title(f"per-point f_r estimates: {int(used.sum())} of {len(x)} used", fontsize=8)
    ax.legend(fontsize=6, loc="upper left", framealpha=0.8)


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
    plot_theory(ax, ds, f[zz[0]], f[zz[-1]], numbers=False)
    th = ds.get("theory")
    if th is not None:
        for i, ft in enumerate(th):
            if f[zz[0]] <= ft <= f[zz[-1]]:
                ax.annotate(f"analytical #{i + 1}: {ft:.1f} Hz", (ft, 0.02),
                            xycoords=("data", "axes fraction"), rotation=90, fontsize=7,
                            color=THEORY_COLOR, ha="right", va="bottom")
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
    plot_theory(ax, ds, f_lo, f_hi)
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
            "[ ] = band -/+   , . = move fit centre one bin   shift+click = no snapping   "
            "u = undo   \u2190 \u2192 = browse steps\n"
            "Point slider, \u2191 \u2193, or click a point on the mode shape or estimates = show its circle   "
            "f = model FRF window   t = analytical tuning window   s = save   q = save & quit\n"
            "scroll = zoom any plot (shift: x only, ctrl: y only)   0 = reset zoom of plot under "
            "the mouse (elsewhere: all)   o / p = toolbar box zoom / pan")

    def __init__(self, sess, outdir):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider
        self.plt = plt
        self.s, self.outdir = sess, outdir
        self.proposal = None
        self.sel = None             # point index chosen for the left circle (None = strongest)
        self.cur = None             # arguments of the fit currently drawn at the bottom
        self.diag = None            # point diagnostics for that fit
        self.step = None            # None = live (latest); else number of modes shown
        self.ctx = {}               # axes -> what it currently shows (for remembering zoom)
        self.saved_lims = {}        # (axes id, context) -> (xlim, ylim)
        self.fig = plt.figure(figsize=(16, 9.5))
        gs = self.fig.add_gridspec(2, 4, height_ratios=[1.15, 1])
        self.ax_top = self.fig.add_subplot(gs[0, :])
        self.ax_ny = [self.fig.add_subplot(gs[1, 0])]
        self.ax_est = [self.fig.add_subplot(gs[1, 1])]
        self.ax_zoom = self.fig.add_subplot(gs[1, 2])
        self.ax_shape = self.fig.add_subplot(gs[1, 3])
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        for k in ("keymap.save", "keymap.quit", "keymap.back", "keymap.forward",
                  "keymap.xscale", "keymap.yscale"):      # k / l would toggle log axes
            plt.rcParams[k] = []
        plt.rcParams["keymap.fullscreen"] = ["ctrl+f"]    # free up f
        self.set_title(self.fig, "circle fit")
        f = sess.f
        self.view = (f >= sess.fmin * 0.8) & (f <= sess.fmax)
        self.ax_top.set_xscale("log")
        self.draw_top(keep_limits=False)
        self.clear_bottom("Click a peak in the top panel (or press n) to fit the next mode.")
        self.ax_shape.set_xlabel(" \n \n ", fontsize=7)    # reserve room for the highlight list
        self.fig.tight_layout(rect=[0, 0.045, 1, 0.93])

        # step-browser buttons (top right)
        self.btn_prev = Button(self.fig.add_axes([0.80, 0.945, 0.055, 0.035]), "\u25c0 Prev")
        self.btn_next = Button(self.fig.add_axes([0.86, 0.945, 0.055, 0.035]), "Next \u25b6")
        self.btn_live = Button(self.fig.add_axes([0.92, 0.945, 0.06, 0.035]), "Latest")
        self.btn_prev.on_clicked(lambda e: self.browse(-1))
        self.btn_next.on_clicked(lambda e: self.browse(+1))
        self.btn_live.on_clicked(lambda e: self.go_live())
        # point slider under the two circle panels, plus a reset button
        labels = sess.ds["labels"]
        self.slider = Slider(self.fig.add_axes([0.08, 0.012, 0.33, 0.022]), "Point",
                             min(labels), max(labels), valinit=labels[0],
                             valstep=np.array(labels, float), valfmt="%d")
        self.slider.on_changed(self.on_slider)
        self.btn_strong = Button(self.fig.add_axes([0.445, 0.006, 0.065, 0.034]), "Strongest")
        self.btn_strong.on_clicked(lambda e: self.select_point(None))

        self.fig2 = None
        self.fig3 = None
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
        self.fig2.canvas.mpl_connect("scroll_event", self.on_scroll)
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
                     "frequencies in Hz; use the toolbar zoom to look closely.", fontsize=10, pad=12)
        if lims:
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])
        self.fig2.canvas.draw_idle()

    # -- third window: tune the analytical plate against the fitted peaks -----
    TUNE_SPECS = [  # key, label, min, max, to slider, from slider
        ("t", "thickness (mm)", 0.4, 1.4, lambda v: v, lambda v: v),
        ("L", "length L, perpendicular to clamp (mm)", 100, 200, lambda v: v, lambda v: v),
        ("W", "width W, along clamp (mm)", 100, 200, lambda v: v, lambda v: v),
        ("c", "clamp depth (mm)", 0, 40, lambda v: v, lambda v: v),
        ("E", "Young's modulus E (GPa)", 40, 100, lambda v: v / 1e9, lambda v: v * 1e9),
        ("rho", "density (kg/m\u00b3)", 2300, 3100, lambda v: v, lambda v: v),
        ("nu", "Poisson's ratio", 0.2, 0.45, lambda v: v, lambda v: v),
        ("k", "clamp stiffness log10 k (N\u00b7m/rad/m); right end = rigid", 0.0, 6.0,
         lambda v: 6.0 if v is None else float(np.clip(np.log10(v), 0, 6)),
         lambda v: None if v >= 5.99 else 10.0**v),
    ]

    def open_tune_window(self):
        from matplotlib.widgets import Slider, Button
        ds = self.s.ds
        self.tune = dict(ds.get("plate") or DEFAULT_PLATE)
        self.tune_start = dict(self.tune)
        self.tune_pushed = None
        self.fig3 = self.plt.figure(figsize=(15, 8.5))
        self.set_title(self.fig3, "analytical tuning")
        self.ax_tune = self.fig3.add_axes([0.05, 0.47, 0.62, 0.47])
        self.ax_tab = self.fig3.add_axes([0.70, 0.47, 0.29, 0.49])
        self.ax_tab.axis("off")
        self.tune_sliders = {}
        for i, (key, lab, lo, hi, to_s, _) in enumerate(self.TUNE_SPECS):
            ax = self.fig3.add_axes([0.30, 0.355 - i * 0.04, 0.33, 0.024])
            sl = Slider(ax, lab, lo, hi, valinit=to_s(self.tune[key]))
            sl.label.set_fontsize(8)
            sl.on_changed(lambda v, key=key: self.on_tune_slider(key, v))
            self.tune_sliders[key] = sl
        from matplotlib.widgets import CheckButtons
        names = {"t": "thickness", "L": "length", "W": "width", "c": "clamp depth",
                 "E": "E", "rho": "density", "nu": "Poisson's ratio", "k": "clamp stiffness"}
        self.tune_keys = [sp[0] for sp in self.TUNE_SPECS]
        cax = self.fig3.add_axes([0.70, 0.11, 0.13, 0.27])
        cax.set_title("vary when optimizing", fontsize=8, loc="left")
        self.tune_checks = CheckButtons(cax, [names[k] for k in self.tune_keys],
                                        [k in ("E", "k") for k in self.tune_keys])
        for t in self.tune_checks.labels:
            t.set_fontsize(8)
        self.tune_band = None       # (lo, hi) Hz to score/optimize over; None = all
        self.span = None
        self.btn_tune_all = Button(self.fig3.add_axes([0.85, 0.405, 0.13, 0.04]),
                                   "All frequencies")
        self.btn_tune_all.on_clicked(lambda e: self.set_band(None))
        from matplotlib.widgets import TextBox
        self.band_box = TextBox(self.fig3.add_axes([0.765, 0.41, 0.065, 0.035]),
                                "band (Hz) ", initial="all")
        self.band_box.label.set_fontsize(8)
        self.band_box.on_submit(self.on_band_text)
        self.btn_tune_opt = Button(self.fig3.add_axes([0.85, 0.335, 0.13, 0.055]), "Optimize")
        self.btn_tune_opt.on_clicked(lambda e: self.optimize_tune())
        self.btn_tune_reset = Button(self.fig3.add_axes([0.85, 0.265, 0.13, 0.045]), "Reset")
        self.btn_tune_reset.on_clicked(lambda e: self.reset_tune())
        self.btn_tune_print = Button(self.fig3.add_axes([0.85, 0.205, 0.13, 0.045]),
                                     "Print settings")
        self.btn_tune_print.on_clicked(
            lambda e: print("  analytical settings:\n    " + plate_cli(self.tune)))
        self.tune_status = self.fig3.text(0.85, 0.195, "", fontsize=8, color="C2", va="top",
                                          wrap=True)
        self.fig3.text(0.70, 0.012, "Drag across the plot or type a band (e.g. 150-700) to choose the\n"
                       "frequencies to score and optimize; peaks outside are faded and ignored.\n"
                       "Lines follow the sliders; other windows update when you\n"
                       "release the mouse. Peaks = your accepted modes (or detected peaks\n"
                       "if none). Pairing is one-to-one by frequency, not by mode shape.",
                       fontsize=7.5, color="0.3", va="bottom")
        self.fig3.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig3.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig3.canvas.mpl_connect("button_release_event", lambda e: self.push_tune())
        self.fig3.canvas.mpl_connect("close_event", self.on_tune_closed)
        self.update_k_text()
        self.draw_tune(keep_limits=False)

    def optimize_tune(self):
        free = [k for k, on in zip(self.tune_keys, self.tune_checks.get_status()) if on]
        meas = self.band_peaks()
        if not free or len(meas) < 2:
            self.tune_status.set_text("Tick at least one parameter, and have at\n"
                                      "least two accepted modes in the band.")
            self.fig3.canvas.draw_idle()
            return
        self.tune_status.set_text("optimizing...")
        self.fig3.canvas.draw()
        self.fig3.canvas.flush_events()
        q, before, after = optimize_plate(self.tune, meas, self.TUNE_SPECS, free, self.s.fmax)
        self.tune = q
        for key, *rest in self.TUNE_SPECS:
            sl = self.tune_sliders[key]
            sl.eventson = False
            sl.set_val(rest[3](q[key]))
            sl.eventson = True
        self.update_k_text()
        self.draw_tune()
        self.push_tune()
        shown = ", ".join(f"{k}" for k in free)
        note = ""
        if sum(k in free for k in ("t", "E", "rho")) > 1:
            note = "\nnote: thickness, E and density only\nmatter together, so their split\nis arbitrary"
        hit = [k for k in free if self.tune_sliders[k].val in
               (self.tune_sliders[k].valmin, self.tune_sliders[k].valmax)]
        if hit:
            note += "\nat slider limit: " + ", ".join(hit)
        band = ("all peaks" if self.tune_band is None
                else f"{self.tune_band[0]:.0f}\u2013{self.tune_band[1]:.0f} Hz ({len(meas)} peaks)")
        self.tune_status.set_text(f"rms {before:.1f}% \u2192 {after:.1f}%\nvaried: {shown}\n"
                                  f"band: {band}{note}")
        print(f"  optimized ({shown}): rms {before:.1f}% -> {after:.1f}%\n    " + plate_cli(q))
        self.fig3.canvas.draw_idle()

    def band_peaks(self):
        meas = self.measured_peaks()
        if self.tune_band is None:
            return meas
        lo, hi = self.tune_band
        return [f for f in meas if lo <= f <= hi]

    def set_band(self, band):
        self.tune_band = band
        if getattr(self, "band_box", None) is not None:      # keep the typed box in step
            self.band_box.eventson = False
            self.band_box.set_val("all" if band is None else f"{band[0]:.0f}-{band[1]:.0f}")
            self.band_box.eventson = True
        self.draw_tune()

    def on_band_text(self, text):
        t = text.strip().lower()
        if t in ("", "all"):
            self.set_band(None)
            return
        nums = re.findall(r"\d+(?:\.\d*)?", t)
        if len(nums) == 2 and float(nums[0]) != float(nums[1]):
            lo, hi = sorted(float(n) for n in nums)
            self.set_band((lo, hi))
        else:
            self.tune_status.set_text("Type a band as two frequencies,\ne.g. 150-700, or 'all'.")
            self.fig3.canvas.draw_idle()

    def on_band(self, lo, hi):
        if hi / max(lo, 1e-9) < 1.005:         # a plain click, not a drag
            return
        self.set_band((lo, hi))

    def on_tune_closed(self, ev):
        self.fig3 = None

    def update_k_text(self):
        sl = self.tune_sliders["k"]
        sl.valtext.set_text("rigid" if self.tune["k"] is None else f"{self.tune['k']:.3g}")

    def on_tune_slider(self, key, v):
        spec = next(sp for sp in self.TUNE_SPECS if sp[0] == key)
        self.tune[key] = spec[5](v)
        if key == "k":
            self.update_k_text()
        self.draw_tune()

    def reset_tune(self):
        self.tune = dict(self.tune_start)
        for key, *_rest in self.TUNE_SPECS:
            sl = self.tune_sliders[key]
            sl.eventson = False
            sl.set_val(_rest[3](self.tune[key]))
            sl.eventson = True
        self.update_k_text()
        self.draw_tune()
        self.push_tune()

    def measured_peaks(self):
        s = self.s
        if s.modes:
            return sorted(m["fr"] for m in s.modes)
        from scipy.signal import find_peaks
        sel = s.band
        pk, _ = find_peaks(mif_db(s.ds["H"][:, sel]), prominence=8)
        return list(s.f[sel][pk][:30])

    def draw_tune(self, keep_limits=True):
        if self.fig3 is None:
            return
        s, ds, ax = self.s, self.s.ds, self.ax_tune
        all_meas = self.measured_peaks()
        meas = self.band_peaks()
        hi_f = min(s.fmax, 1.35 * max(all_meas)) if all_meas else s.fmax
        th = plate_theory(self.tune, s.fmax, n_terms=16, n_max=80)
        lims = (ax.get_xlim(), ax.get_ylim()) if keep_limits else None
        ax.clear()
        v = (s.f >= s.fmin * 0.8) & (s.f <= s.fmax)
        ax.plot(s.f[v], mif_db(ds["H"][:, v]), color="0.6", lw=0.7, label="measured")
        for i, fm in enumerate(all_meas):
            inside = fm in meas
            ax.axvline(fm, color="C3", lw=1.2, alpha=0.8 if inside else 0.2,
                       label="your peaks" if i == 0 else None)
        if self.tune_band is not None:
            ax.axvspan(*self.tune_band, color="C2", alpha=0.08, label="optimization band")
        plot_theory(ax, {"theory": th, "theory_label": "analytical (sliders)"}, s.fmin * 0.8,
                    s.fmax)
        pairs = match_modes(meas, th)
        import matplotlib.transforms as mtransforms
        tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for fm, j, fa, err in pairs:      # arrow from each peak to its nearest analytical mode
            ax.annotate("", xy=(fa, 0.93), xytext=(fm, 0.93), xycoords=tr, textcoords=tr,
                        arrowprops=dict(arrowstyle="->", color="C3", lw=1))
        ax.set_xscale("log")
        ax.set_xlabel("Hz")
        ax.set_ylabel("rms |H| (dB)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="lower right", fontsize=8)
        errs = np.array([p[3] for p in pairs])
        head = (f"mean {errs.mean():+.1f}%, rms {np.sqrt(np.mean(errs**2)):.1f}%"
                if len(errs) else "no peaks to compare")
        band = ("" if self.tune_band is None
                else f" in {self.tune_band[0]:.0f}\u2013{self.tune_band[1]:.0f} Hz")
        ax.set_title(f"analytical vs your peaks{band}: {head}\n{plate_label(self.tune)}",
                     fontsize=9, pad=14)
        ax.set_xlim(lims[0] if lims else (s.fmin * 0.8, hi_f))
        if lims:
            ax.set_ylim(lims[1])
        # table
        self.ax_tab.clear()
        self.ax_tab.axis("off")
        used = [p[1] for p in pairs]
        rows = ["peak (Hz)   analytical #   (Hz)   diff"]
        for fm, j, fa, err in pairs:
            dup = " *" if used.count(j) > 1 else ""
            rows.append(f"{fm:9.1f}   {j + 1:6d}   {fa:9.1f}   {err:+6.1f}%{dup}")
        if len(all_meas) > len(meas):
            rows.append(f"({len(all_meas) - len(meas)} peaks outside the band not shown)")
        if len(pairs) < len(meas):
            rows.append(f"({len(meas) - len(pairs)} peaks above the highest analytical mode)")
        self.ax_tab.text(0, 1, "\n".join(rows), family="monospace", fontsize=8, va="top")
        # drag-to-select band; recreated because clearing the axes removes its artists
        from matplotlib.widgets import SpanSelector
        if self.span is not None:
            self.span.disconnect_events()
        self.span = SpanSelector(ax, self.on_band, "horizontal", useblit=False,
                                 props=dict(facecolor="C2", alpha=0.2))
        self.fig3.canvas.draw_idle()

    def push_tune(self):
        """Send the tuned analytical frequencies to the other windows."""
        if self.fig3 is None or self.tune_pushed == self.tune:
            return
        self.tune_pushed = dict(self.tune)
        ds = self.s.ds
        ds["plate"] = dict(self.tune)
        ds["theory"] = plate_theory(self.tune, self.s.fmax)
        ds["theory_label"] = plate_label(self.tune)
        self.draw_top()
        if self.cur is not None:
            self.draw_fit(*self.cur)
        self.fig.canvas.draw_idle()
        self.draw_model()

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
        plot_theory(ax, ds, f[0], f[-1])
        if self.browsing() and n:
            ax.axvline(s.modes[n - 1]["fr"], color="r", ls="--", lw=0.9)
        if self.proposal:
            p = self.proposal
            zz = np.arange(max(p["idx"][0] - 20, 1), min(p["idx"][-1] + 20, len(s.f) - 1))
            ax.plot(s.f[zz], mif_db(s.R[:, zz] - contribution(p, s.w[zz])), color="C2",
                    lw=1.2, label="residual after proposed mode")
            ax.axvline(p["fr"], color="r", ls="--", lw=0.8)
            ax.axvline(p["peak_hz"], color="0.3", ls=":", lw=0.9,
                       label=f"fit centre bin {p['peak_hz']:.2f} Hz")
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
        ax.set_title(f"{ds['name']}{' (sign flipped)' if ds.get('flipped') else ''}:  {head},  residual {res:.2f} dB re measured "
                     f"(mean over {s.fmin:g}\u2013{s.fmax:g} Hz)", fontsize=10, color=color, pad=12)
        if lims:
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])
        self.fig.suptitle(self.HELP, fontsize=9, color="0.3", x=0.34)

    def clear_bottom(self, msg):
        self.cur = None
        for a in self.bottom_axes():
            self.remember(a)
            self.ctx.pop(a, None)
            a.clear()
            a.set_xticks([])
            a.set_yticks([])
        self.ax_zoom.text(0.5, 0.5, msg, ha="center", va="center", wrap=True,
                          transform=self.ax_zoom.transAxes, fontsize=9)

    def draw_fit(self, m, R_band, R_zz, zz, title, color):
        self.cur = (m, R_band, R_zz, zz, title, color)
        ds = self.s.ds
        self.diag = point_diagnostics(m, R_band, ds["f"][m["idx"]])
        self.remember(self.ax_zoom)
        self.ax_zoom.clear()
        self.draw_points()
        plot_zoom(self.ax_zoom, ds, m, R_zz, zz)
        self.recall(self.ax_zoom, m["uid"])
        self.ax_zoom.set_title(f"{title}: {m['fr']:.3f} Hz, eta {m['eta']:.5f}, "
                               f"{len(m['idx'])} bins, {m['gain_db']:.1f} dB better",
                               fontsize=8, color=color)

    def draw_points(self):
        """Circle panel, estimates panel and mode shape (all that changes with the point)."""
        m, R_band = self.cur[0], self.cur[1]
        ds, diag = self.s.ds, self.diag
        for a in self.ax_ny + self.ax_est + [self.ax_shape]:
            self.remember(a)
            a.clear()
        left = int(np.argmax(np.abs(m["A"]))) if self.sel is None else self.sel
        a = self.ax_ny[0]
        plot_nyquist(a, ds, m, R_band, left, legend=True)
        st = {0: "used", 1: "excluded: small circle", 2: "excluded: poor circle",
              3: "excluded: no circle"}
        status = st[int(m["est"]["status"][left])] if "est" in m else ""
        a.set_title(f"point {ds['labels'][left]} "
                    f"({'your pick' if self.sel is not None else 'strongest'}, {status})\n"
                    + describe_point(ds, diag, left), fontsize=8,
                    color="C0" if self.sel is not None else "k")
        plot_estimates(self.ax_est[0], ds, m, selected=left)
        plot_shape(self.ax_shape, ds, m["A"],
                   "mode shape (real part)  \u25a1 = selected point", diag, selected=left)
        self.recall(self.ax_ny[0], (m["uid"], left))      # circle zoom is per point
        self.recall(self.ax_est[0], m["uid"])
        self.recall(self.ax_shape, m["uid"])
        # keep the slider showing the point on display, without re-triggering it
        self.slider.eventson = False
        self.slider.set_val(ds["labels"][left])
        self.slider.eventson = True

    # -- zoom -------------------------------------------------------------
    def bottom_axes(self):
        return self.ax_ny + self.ax_est + [self.ax_zoom, self.ax_shape]

    def data_axes(self, fig):
        if fig is self.fig:
            return [self.ax_top] + self.bottom_axes()
        if self.fig2 is not None and fig is self.fig2:
            return [self.ax_model]
        if self.fig3 is not None and fig is self.fig3:
            return [self.ax_tune]
        return []

    def remember(self, ax):
        """Store ax's current view under what it shows, before it is redrawn."""
        if ax in self.ctx:
            self.saved_lims[(id(ax), self.ctx[ax])] = (ax.get_xlim(), ax.get_ylim())

    def recall(self, ax, ctx):
        """ax now shows ctx; restore the view it had last time it showed ctx."""
        self.ctx[ax] = ctx
        lims = self.saved_lims.get((id(ax), ctx))
        if lims:
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])

    def reset_zoom(self, ax):
        if ax in self.ctx:
            self.saved_lims.pop((id(ax), self.ctx.pop(ax)), None)
        if ax is self.ax_top:
            self.draw_top(keep_limits=False)
        elif self.fig2 is not None and ax is self.ax_model:
            self.draw_model(keep_limits=False)
        elif self.fig3 is not None and ax is self.ax_tune:
            self.draw_tune(keep_limits=False)
        elif self.cur is not None and ax in self.bottom_axes():
            self.draw_fit(*self.cur)

    def on_scroll(self, ev):
        ax = ev.inaxes
        if ax is None or ax not in self.data_axes(ax.figure) or ev.xdata is None:
            return
        f = 1 / 1.25 if ev.button == "up" else 1.25          # up = zoom in
        mods = ev.key or ""

        def zoom(lims, c, log):
            if log:
                lims, c = np.log10(lims), np.log10(c)
            new = c + (np.array(lims) - c) * f
            return 10**new if log else new

        if "control" not in mods and "ctrl" not in mods:
            ax.set_xlim(zoom(ax.get_xlim(), ev.xdata, ax.get_xscale() == "log"))
        if "shift" not in mods:
            ax.set_ylim(zoom(ax.get_ylim(), ev.ydata, ax.get_yscale() == "log"))
        ax.figure.canvas.draw_idle()

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

    def select_point(self, p):
        """Show point index p (None = strongest) in the left circle panel."""
        self.sel = p
        if self.cur is not None:
            self.draw_points()
            self.fig.canvas.draw_idle()

    def on_slider(self, val):
        labels = self.s.ds["labels"]
        p = int(np.argmin(np.abs(np.array(labels) - val)))
        if p != self.sel:
            self.select_point(p)

    def step_point(self, d):
        labels = self.s.ds["labels"]
        cur = labels.index(int(round(self.slider.val)))
        self.slider.set_val(labels[min(max(cur + d, 0), len(labels) - 1)])

    def on_click(self, ev):
        tb = getattr(self.fig.canvas, "toolbar", None)
        if tb is not None and str(getattr(tb, "mode", "")):
            return                      # zoom / pan active
        if ev.button != 1 or ev.xdata is None:
            return
        if ev.inaxes is self.ax_top:
            # shift+click: use exactly the clicked bin; plain click snaps to the nearest peak
            self.fit_at(ev.xdata, snap="shift" not in (ev.key or ""))
        elif ev.inaxes in self.ax_est and self.cur is not None:
            self.select_point(int(np.argmin(np.abs(np.array(self.s.ds["labels"]) - ev.xdata))))
        elif ev.inaxes is self.ax_shape and self.cur is not None:
            ds = self.s.ds
            if ds["xy"] is not None:
                p = int(np.argmin(np.hypot(ds["xy"][:, 0] - ev.xdata,
                                           ds["xy"][:, 1] - ev.ydata)))
            else:
                p = int(np.argmin(np.abs(np.array(ds["labels"]) - ev.xdata)))
            self.select_point(p)

    def on_key(self, ev):
        box = getattr(self, "band_box", None)
        if box is not None and box.capturekeystrokes:
            return                      # typing in the band box
        k = ev.key
        if k in ("up", "down"):
            self.step_point(+1 if k == "up" else -1)
        elif k == "left":
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
        elif k in (",", ".") and self.proposal:
            p = self.proposal
            self.fit_at(p["peak_hz"] + (1 if k == "." else -1) * self.s.ds["df"],
                        side=p["side"], snap=False)
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
        elif k == "0":
            fig = ev.canvas.figure
            targets = ([ev.inaxes] if ev.inaxes in self.data_axes(fig)
                       else self.data_axes(fig))
            for a in targets:
                if a in self.bottom_axes():
                    if a in self.ctx:
                        self.saved_lims.pop((id(a), self.ctx.pop(a)), None)
                else:
                    self.reset_zoom(a)
            if any(a in self.bottom_axes() for a in targets) and self.cur is not None:
                self.draw_fit(*self.cur)
            fig.canvas.draw_idle()
        elif k == "t":
            if self.fig3 is None:
                self.open_tune_window()
                self.fig3.show()
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
            for fg in (self.fig2, self.fig3):
                if fg is not None:
                    self.plt.close(fg)
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
        order = np.argsort(-np.abs(m["A"]))[:2]
        off = m["idx"][0] - m["zz"][0]
        Rb_band = m["R_before_zz"][:, off:off + len(m["idx"])]
        for j, p in enumerate(order):
            plot_nyquist(fig.add_subplot(gs[0, j]), ds, m, Rb_band, p, legend=(j == 0))
        plot_estimates(fig.add_subplot(gs[0, 2]), ds, m)
        plot_zoom(fig.add_subplot(gs[1, :2]), ds, m, m["R_before_zz"], m["zz"])
        plot_shape(fig.add_subplot(gs[1, 2]), ds, m["A"], "mode shape (real part)",
                   point_diagnostics(m, Rb_band, ds["f"][m["idx"]]))
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
    plot_theory(ax, ds, f[v][0], f[v][-1])
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
                 "(labels = natural frequencies, Hz)", pad=12)
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
    ap.add_argument("--plate", default="150,150,0.86,15",
                    help="analytical cantilever plate for comparison: length,width,thickness,"
                         "clamp depth in mm (length is perpendicular to the clamped edge; "
                         "the free span is length - clamp). 'none' turns it off. "
                         "Default 150,150,0.86,15 (measured plate thickness 0.86 mm)")
    ap.add_argument("--flip-sign", action="store_true",
                    help="multiply all FRFs by -1, e.g. when the accelerometer is mounted upside "
                         "down relative to the hammer direction, so circles at points moving in "
                         "phase with the accelerometer hang down")
    ap.add_argument("--clamp-k", default="rigid",
                    help="clamp rotational stiffness for the analytical plate, N*m/rad per metre "
                         "of clamped edge, or 'rigid' (default)")
    ap.add_argument("--E", type=float, default=69e9, help="Young's modulus, Pa (default Al, 69e9)")
    ap.add_argument("--nu", type=float, default=0.33, help="Poisson's ratio (default 0.33)")
    ap.add_argument("--rho", type=float, default=2700.0, help="density, kg/m^3 (default 2700)")
    args = ap.parse_args()

    if args.no_gui:
        matplotlib.use("Agg")
    ds = load_dataset(args.folder)
    outdir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "CircleFit", ds["name"])
    print(f"{ds['name']}: {len(ds['labels'])} points, df = {ds['df']:.4f} Hz")
    ds["flipped"] = args.flip_sign
    if args.flip_sign:
        ds["H"] = -ds["H"]
        print("  sign flipped: all FRFs multiplied by -1")
    ds["theory"] = None
    pl = dict(DEFAULT_PLATE)
    if args.plate.lower() != "none":
        pl["L"], pl["W"], pl["t"], pl["c"] = (float(v) for v in args.plate.split(","))
    pl.update(E=args.E, nu=args.nu, rho=args.rho,
              k=None if args.clamp_k.lower() == "rigid" else float(args.clamp_k))
    ds["plate"] = pl
    if args.plate.lower() != "none":
        ds["theory"] = plate_theory(pl, args.fmax)
        ds["theory_label"] = plate_label(pl)
        print(f"  {ds['theory_label']}: {len(ds['theory'])} modes up to {args.fmax:g} Hz")
        print("  " + ", ".join(f"{i + 1}: {v:.1f}" for i, v in enumerate(ds["theory"])))
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

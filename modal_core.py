"""
modal_core.py: the computational core of the incremental circle fit, shared by
the desktop script (circle_fit_modes.py) and the web app (index.html), which
runs this same file in the browser through Pyodide.

Contents: ObieApp .trf / test-folder loading, per-point and global circle
fitting, residual bookkeeping (Session), the analytical cantilever-plate model
(Rayleigh-Ritz, rigid or elastic clamp), peak pairing and parameter
optimization, per-point diagnostics and the result tables.

Only numpy and scipy are needed (no matplotlib).
"""
import csv
import functools
import glob
import io
import itertools
import json
import os
import re
import struct
import sys

import numpy as np
from scipy.optimize import least_squares

HEADER_BYTES = 110
TRF_HEADER = "<I4d2s3d11f4s"   # ObieApp trf_fileio.py header layout (110 bytes)
_FIT_IDS = itertools.count(1)   # unique id per fit, so saved zoom never leaks between fits
MAX_ETA = 0.2      # upper bound on loss factor (avoids circle/offset degeneracy)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def read_trf(path_or_bytes):
    """Read an ObieApp .trf file (path or raw bytes). Returns (df, H, coherence).

    Layout (little-endian, as in ObieApp's trf_fileio.py): a 110-byte header
    '<I4d2s3d11f4s' holding Hz_Resolution, Start_Freq, End_Freq, fComplex and
    fLength, then fLength bins of (re, im) or (re, im, coherence) float64, then
    an optional '\\0OBIE_META' text block. Circle fitting needs complex data,
    so magnitude-only files (fComplex 0) are rejected. Bins are assumed to
    start at 0 Hz (Start_Freq is checked)."""
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        b, name = bytes(path_or_bytes), "data"
    else:
        name = path_or_bytes
        with open(path_or_bytes, "rb") as fh:
            b = fh.read()
    if len(b) < HEADER_BYTES:
        raise ValueError(f"{name}: file too short for a .trf header")
    h = struct.unpack(TRF_HEADER, b[:HEADER_BYTES])
    df, f0, fcomplex, n = h[6], h[7], round(h[9]), int(h[10])
    per_bin = {1: 2, 2: 3}.get(fcomplex)
    if per_bin is None:
        raise ValueError(f"{name}: magnitude-only .trf (fComplex={h[9]}); circle fitting "
                         "needs complex FRFs")
    if not (0 < df < 100) or n <= 0 or len(b) < HEADER_BYTES + 8 * per_bin * n:
        raise ValueError(f"{name}: unrecognised .trf header (df={df}, n={n})")
    if abs(f0) > 1e-9:
        raise ValueError(f"{name}: Start_Freq {f0} Hz; only 0 Hz is supported")
    a = np.frombuffer(b[HEADER_BYTES:HEADER_BYTES + 8 * per_bin * n], "<f8").reshape(n, per_bin)
    coh = a[:, 2] if per_bin == 3 else np.ones(n)
    return df, a[:, 0] + 1j * a[:, 1], coh


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

    # coarse grid search for a robust start (all 60 x 50 candidates at once),
    # then least-squares polish
    WR = np.linspace(w[0], w[-1], 60)[:, None, None]
    ETA = np.geomspace(1e-4, MAX_ETA, 50)[None, :, None]
    M = s * 2 * np.arctan((1 - w2[None, None, :] / WR**2) / ETA)
    phi0 = np.angle(np.mean(np.exp(1j * (phi - M)), axis=-1, keepdims=True))
    cost = np.sum(_wrap(phi - M - phi0) ** 2, axis=-1)
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    wr0, eta0 = WR[i, 0, 0], ETA[0, j, 0]
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
        k = int(np.clip(round(f_click / self.ds["df"]), 1, len(self.f) - 2))
        lo, hi = max(k - search_bins - 1, 0), min(k + search_bins + 1, len(self.f) - 1)
        lv = rms_level(self.R[:, lo:hi + 1])
        c = k - lo
        for d in range(search_bins + 1):
            for j in (c - d, c + d):
                if 1 <= j < len(lv) - 1 and lv[j] >= lv[j - 1] and lv[j] >= lv[j + 1]:
                    return lo + j
        return lo + 1 + int(np.argmax(lv[1:-1]))

    def propose(self, f_click, side=None, snap=True):
        if not 0 < f_click < self.f[-1]:
            return None
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


PLATE_SPECS = [  # key, label, min, max, to slider units, from slider units
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


# ----------------------------------------------------------------------------
# Per-point helpers
# ----------------------------------------------------------------------------
def real_shape(A):
    """Rotate the complex mode shape so it is as real as possible."""
    ang = 0.5 * np.angle(np.sum(A**2))
    s = (A * np.exp(-1j * ang)).real
    return s / np.max(np.abs(s))


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


def mode_title(m):
    return (f"f_r = {m['fr']:.3f} Hz   eta = {m['eta']:.5f}  (Q = {1/m['eta']:.0f})   "
            f"[circle-only {m['fr_circle']:.3f} Hz, eta {m['eta_circle']:.5f}]   "
            f"band {len(m['idx'])} bins   local improvement {m['gain_db']:.1f} dB")


def parse_modes(text):
    out = []
    for tok in filter(None, (t.strip() for t in text.split(","))):
        fhz, _, side = tok.partition(":")
        out.append((float(fhz), int(side) if side else None))
    return out


# ----------------------------------------------------------------------------
# Result tables (shared by the desktop script and the web app)
# ----------------------------------------------------------------------------
MODES_HEADER = ["mode", "f_r_Hz", "eta", "zeta", "Q", "f_r_circle_only_Hz",
                "eta_circle_only", "band_lo_Hz", "band_hi_Hz", "peak_Hz",
                "half_width_bins", "n_points_used_for_circle_estimate",
                "local_improvement_dB", "residual_after_dB_re_measured"]


def modes_table(sess):
    ds = sess.ds
    rows = [MODES_HEADER]
    for m in sess.modes:
        rows.append([m["n"], f"{m['fr']:.4f}", f"{m['eta']:.6f}", f"{m['eta']/2:.6f}",
                     f"{1/m['eta']:.1f}", f"{m['fr_circle']:.4f}",
                     f"{m['eta_circle']:.6f}", f"{ds['f'][m['idx'][0]]:.2f}",
                     f"{ds['f'][m['idx'][-1]]:.2f}", f"{m['peak_hz']:.4f}",
                     m["side"], m["n_circles"], f"{m['gain_db']:.2f}",
                     f"{m['residual_after']:.3f}"])
    return rows


def shapes_table(sess):
    ds, modes = sess.ds, sess.modes
    hdr = ["point", "x_mm", "y_mm"]
    for m in modes:
        hdr += [f"m{m['n']}_{m['fr']:.1f}Hz_real", f"m{m['n']}_absA", f"m{m['n']}_phaseA_deg"]
    rows = [hdr]
    shapes = [real_shape(m["A"]) for m in modes]
    for p, lab in enumerate(ds["labels"]):
        row = [lab] + (list(ds["xy"][p]) if ds["xy"] is not None else ["", ""])
        for m, s in zip(modes, shapes):
            row += [f"{s[p]:.5f}", f"{abs(m['A'][p]):.6g}",
                    f"{np.degrees(np.angle(m['A'][p])):.2f}"]
        rows.append(row)
    return rows


def table_csv(rows):
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerows(rows)
    return out.getvalue()


def replay_string(sess):
    """The modes as 'peak_Hz:half_width' tokens, the --modes replay format."""
    return ",".join(f"{m['peak_hz']:.4f}:{m['side']}" for m in sess.modes)

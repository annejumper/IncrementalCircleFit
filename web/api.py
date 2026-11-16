"""
api.py: the web app's bridge to modal_core.py. Runs inside Pyodide in a web
worker; every function takes plain arguments and returns a JSON string that
the page (web/app.js) turns into Plotly figures. All fitting maths lives in
modal_core.py, shared with the desktop script.
"""
import json

import numpy as np

import modal_core as mc

S = None            # mc.Session
PROPOSAL = None     # current proposed mode (not yet accepted)
VIEW = None         # boolean mask of the bins shown in the top plot


# ----------------------------------------------------------------------------
def _r(a, nd=3):
    """Round an array to a JSON-friendly list (NaN/inf -> None)."""
    a = np.asarray(a, float)
    out = np.round(a, nd).tolist()
    if not np.all(np.isfinite(a)):
        out = [None if not np.isfinite(v) else r for v, r in zip(a.ravel(), out)]
    return out


def _js(v):
    """Lists and objects arrive from the page as JSON text."""
    return json.loads(v) if isinstance(v, str) else v


def _db(R):
    return _r(mc.mif_db(R), 2)


def _mode_info(m):
    return dict(n=m.get("n"), uid=m["uid"], fr=m["fr"], eta=m["eta"], Q=1 / m["eta"],
                fr_circle=m["fr_circle"], eta_circle=m["eta_circle"], gain=m["gain_db"],
                peak_hz=m["peak_hz"], side=int(m["side"]), bins=len(m["idx"]),
                residual_after=m.get("residual_after"), title=mc.mode_title(m))


def _set_theory(pl):
    ds = S.ds
    ds["plate"] = pl
    ds["theory"] = mc.plate_theory(pl, S.fmax) if pl is not None else None


def _plate(js):
    if js is None:
        return None
    pl = dict(mc.DEFAULT_PLATE)
    pl.update(json.loads(js) if isinstance(js, str) else js)
    return pl


# ----------------------------------------------------------------------------
def load(folder, fmin, fmax, refine, flip, plate_js, pad_s=0):
    global S, PROPOSAL, VIEW
    ds = mc.load_dataset(folder)
    raw = mc.raw_info(ds, float(fmax))
    if pad_s and float(pad_s) > 0:
        ds = mc.zero_pad_dataset(ds, float(pad_s), float(fmax), raw=raw)
    raw_summary = None
    if raw is not None:
        raw_summary = dict(n=sum(len(v) for v in raw["hits"].values()), record_s=raw["record_s"],
                           error=raw["error"], hammer_channel=raw["hammer_channel"],
                           clipped_response=raw["clipped_response"],
                           clipped_hammer=raw["clipped_hammer"])
    if flip:
        ds["H"] = -ds["H"]
    ds["flipped"] = bool(flip)
    S = mc.Session(ds, float(fmin), float(fmax), bool(refine))
    PROPOSAL = None
    VIEW = (S.f >= S.fmin * 0.8) & (S.f <= S.fmax)
    _set_theory(_plate(plate_js))
    return json.dumps(dict(
        name=ds["name"], n_points=len(ds["labels"]), labels=ds["labels"], df=ds["df"],
        has_xy=ds["xy"] is not None, fmin=S.fmin, fmax=S.fmax,
        coherence_mean=float(np.mean(ds["coh"][:, VIEW])), raw=raw_summary,
        pad_s=ds.get("pad_s"), added_s=ds.get("added_s"), record_s=ds.get("record_s"),
        theory=_r(ds["theory"], 2) if ds["theory"] is not None else None))


def specs():
    """Slider definitions for the analytical plate (ranges in slider units)."""
    names = {"t": "Thickness (mm)", "L": "Length L (mm)", "W": "Width W (mm)",
             "c": "Clamp depth (mm)", "E": "Young's modulus E (GPa)",
             "rho": "Density (kg/m³)", "nu": "Poisson's ratio",
             "k": "Clamp stiffness log₁₀ k (N·m/rad/m)"}
    out = []
    for key, _lab, lo, hi, to_s, _ in mc.PLATE_SPECS:
        out.append(dict(key=key, label=names[key], lo=lo, hi=hi,
                        value=to_s(mc.DEFAULT_PLATE[key])))
    return json.dumps(out)


def residual_shown(step, sel):
    if step is None:
        return S.R[:, sel]
    R = S.ds["H"][:, sel].copy()
    w = S.w[sel]
    for m in S.modes[:step]:
        R -= mc.contribution(m, w)
    return R


def top(step=None, smooth=True, measured=False):
    """Everything the top FRF plot needs for the given step (None = latest).
    The frequency axis and measured curve never change, so they are only sent
    when asked for (once, after loading)."""
    step = None if step is None or step >= len(S.modes) else int(step)
    n = len(S.modes) if step is None else step
    f = S.f[VIEW]
    H = S.ds["H"][:, VIEW]
    Rv = residual_shown(step, VIEW)
    out = dict(residual=_db(Rv), n=n, total=len(S.modes), step=step,
               modes=[_mode_info(m) for m in S.modes])
    if measured:
        out["f"], out["measured"] = _r(f, 4), _db(H)
    out["sum"] = _db(H - Rv) if n else None
    sel = S.band & VIEW
    out["residual_db"] = float(np.mean(mc.mif_db(residual_shown(step, sel))
                                       - mc.mif_db(S.ds["H"][:, sel])))
    if smooth and n:
        ms = S.modes[:n]
        g = np.geomspace(S.fmin * 0.8, S.fmax, 5000)
        ff = np.unique(np.concatenate([g, mc.fine_grid(S.fmin * 0.8, S.fmax, S.ds["df"], ms,
                                                       over=1)[::4]]
                                      + [np.linspace(m["fr"] * (1 - 4 * m["eta"]),
                                                     m["fr"] * (1 + 4 * m["eta"]), 121)
                                         for m in ms]))
        ff = ff[(ff >= S.fmin * 0.8) & (ff <= S.fmax)]
        out["smooth"] = dict(f=_r(ff, 4), db=_r(mc.model_db(ms, ff), 2))
    if PROPOSAL is not None and step is None:
        p = PROPOSAL
        zz = np.arange(max(p["idx"][0] - 20, 1), min(p["idx"][-1] + 20, len(S.f) - 1))
        out["proposal"] = dict(_mode_info(p), f=_r(S.f[zz], 4),
                               after=_db(S.R[:, zz] - mc.contribution(p, S.w[zz])))
    th = S.ds.get("theory")
    out["theory"] = _r(th, 2) if th is not None else None
    return json.dumps(out)


# ----------------------------------------------------------------------------
def propose(f_click, side=None, snap=True):
    global PROPOSAL
    side = None if side is None or side < 0 else int(side)
    p = S.propose(float(f_click), side, bool(snap))
    if p is None:
        return json.dumps(dict(ok=False, msg=f"Could not fit a circle near {f_click:.2f} Hz"))
    PROPOSAL = p
    return json.dumps(dict(ok=True, mode=_mode_info(p)))


def nudge(direction):
    """Move the proposal's centre one bin and refit with the same band."""
    if PROPOSAL is None:
        return json.dumps(dict(ok=False, msg="No proposed fit"))
    return propose(PROPOSAL["peak_hz"] + direction * S.ds["df"], PROPOSAL["side"], False)


def band(delta):
    if PROPOSAL is None:
        return json.dumps(dict(ok=False, msg="No proposed fit"))
    return propose(PROPOSAL["peak_hz"], max(2, PROPOSAL["side"] + int(delta)), False)


def next_peak():
    f = S.next_peak()
    return json.dumps(None if f is None else float(f))


def accept():
    global PROPOSAL
    if PROPOSAL is None:
        return json.dumps(dict(ok=False, msg="No proposed fit to accept"))
    S.accept(PROPOSAL)
    PROPOSAL = None
    return json.dumps(dict(ok=True, mode=_mode_info(S.modes[-1])))


def discard():
    global PROPOSAL
    PROPOSAL = None
    return json.dumps(dict(ok=True))


def undo():
    global PROPOSAL
    m = S.undo()
    PROPOSAL = None
    return json.dumps(dict(ok=m is not None, mode=_mode_info(m) if m else None))


def replay(text):
    global PROPOSAL
    PROPOSAL = None
    done, failed = 0, []
    for fhz, side in mc.parse_modes(text):
        m = S.propose(fhz, side)
        if m is None:
            failed.append(fhz)
            continue
        S.accept(m)
        done += 1
    return json.dumps(dict(done=done, failed=failed))


# ----------------------------------------------------------------------------
def _fit_context(which, n):
    """(mode, residual in the fit band, residual around it, those bins)."""
    if which == "proposal":
        p = PROPOSAL
        if p is None:
            return None
        pad = max(10, 3 * len(p["idx"]))
        zz = np.arange(max(p["idx"][0] - pad, 1), min(p["idx"][-1] + pad, len(S.f) - 1))
        return p, S.R[:, p["idx"]], S.R[:, zz], zz
    m = S.modes[int(n) - 1]
    off = m["idx"][0] - m["zz"][0]
    return m, m["R_before_zz"][:, off:off + len(m["idx"])], m["R_before_zz"], m["zz"]


def fit_view(which, n=None, point=-1):
    """Data for the circle, per-point estimate, zoom and mode-shape panels."""
    ctx = _fit_context(which, n)
    if ctx is None:
        return json.dumps(None)
    m, R_band, R_zz, zz = ctx
    ds, f = S.ds, S.f
    diag = mc.point_diagnostics(m, R_band, f[m["idx"]])
    strongest = int(np.argmax(np.abs(m["A"])))
    p = strongest if point is None or point < 0 else int(point)
    wr = 2 * np.pi * m["fr"]
    w = 2 * np.pi * f[m["idx"]]
    A, B, eta = m["A"][p], m["B"][p], m["eta"]

    # circle: data, fitted arc over the band, full circle, resonance, offset
    z = R_band[p] / (-w**2)
    wf = np.linspace(w[0], w[-1], 300)
    arc = A * mc.mode_basis(wf, wr, eta) + B
    centre = B + A / (2j * eta * wr**2)
    rad = abs(A) / (2 * eta * wr**2)
    th = np.linspace(0, 2 * np.pi, 241)
    full = centre + rad * np.exp(1j * th)
    star = A / (1j * eta * wr**2) + B
    status_names = {0: "used", 1: "excluded: small circle", 2: "excluded: poor circle",
                    3: "excluded: no circle"}
    est = m["est"]
    nyq = dict(point=p, label=ds["labels"][p], strongest=strongest,
               status=status_names[int(est["status"][p])],
               describe=mc.describe_point(ds, diag, p),
               data=[_r(z.real, 12), _r(z.imag, 12)], f=_r(f[m["idx"]], 3),
               arc=[_r(arc.real, 12), _r(arc.imag, 12)],
               full=[_r(full.real, 12), _r(full.imag, 12)],
               star=[float(star.real), float(star.imag)], offset=[float(B.real), float(B.imag)])

    # per-point f_r estimates
    estimates = dict(labels=ds["labels"], dfr=_r(est["fr"] - m["fr"], 4),
                     status=est["status"].astype(int).tolist(),
                     R=_r(est["R"] / max(est["R"].max(), 1e-300), 4),
                     circle_only=m["fr_circle"] - m["fr"], fr=m["fr"], fr_circle=m["fr_circle"],
                     df=ds["df"])

    # zoom around the peak
    fz = f[zz]
    ff = mc.fine_grid(fz[0], fz[-1], ds["df"], [m], over=10)
    c = mc.contribution(m, S.w[zz])
    th_list = ds.get("theory")
    zoom = dict(f=_r(fz, 4), measured=_db(ds["H"][:, zz]), before=_db(R_zz),
                after=_db(R_zz - c), smooth_f=_r(ff, 4), smooth=_r(mc.model_db([m], ff), 2),
                fr=m["fr"], fr_db=float(mc.model_db([m], [m["fr"]])[0]),
                band=[float(f[m["idx"][0]]), float(f[m["idx"][-1]])],
                theory=[[i + 1, float(v)] for i, v in enumerate(th_list)
                        if th_list is not None and fz[0] <= v <= fz[-1]]
                if th_list is not None else [])

    # mode shape
    s = mc.real_shape(m["A"])
    shape = dict(values=_r(s, 4), labels=ds["labels"],
                 picks={k: [int(i) for i in v] for k, v in diag["picks"].items()},
                 selected=p)
    if ds["xy"] is not None:
        x, y = ds["xy"][:, 0], ds["xy"][:, 1]
        shape["x"], shape["y"] = _r(x, 3), _r(y, 3)
        xs, ys = np.unique(x), np.unique(y)
        if len(xs) * len(ys) == len(s):
            grid = np.full((len(ys), len(xs)), np.nan)
            for v, xi, yi in zip(s, x, y):
                grid[np.searchsorted(ys, yi), np.searchsorted(xs, xi)] = v
            shape["grid"] = dict(x=_r(xs, 3), y=_r(ys, 3), z=[_r(r, 4) for r in grid])
    return json.dumps(dict(mode=_mode_info(m), which=which, nyquist=nyq,
                           estimates=estimates, zoom=zoom, shape=shape))


# ----------------------------------------------------------------------------
def _peaks(lo=None, hi=None):
    fs = sorted(m["fr"] for m in S.modes)
    if lo is not None and hi is not None:
        fs = [f for f in fs if lo <= f <= hi]
    return fs


def theory(plate_js, fast=True):
    """Analytical frequencies for these plate settings. fast: a lighter solve
    for live slider feedback (accurate to <0.01% for the first ~30 modes)."""
    pl = _plate(plate_js)
    th = (mc.plate_theory(pl, S.fmax, n_terms=16, n_max=80) if fast
          else mc.plate_theory(pl, S.fmax))
    return json.dumps(dict(theory=_r(th, 2), label=mc.plate_label(pl), cli=mc.plate_cli(pl)))


def theory_from_sliders(values_js, fast=True):
    """Plate settings from slider values, plus their analytical frequencies."""
    pl = json.loads(sliders_to_plate(values_js))
    out = dict(plate=pl, label=mc.plate_label(pl), cli=mc.plate_cli(pl))
    if S is not None:
        out["theory"] = json.loads(theory(json.dumps(pl), fast))["theory"]
    return json.dumps(out)


def set_theory(plate_js):
    """Use these plate settings for the analytical lines everywhere."""
    pl = _plate(plate_js) if plate_js is not None else None
    _set_theory(pl)
    th = S.ds["theory"]
    return json.dumps(dict(theory=_r(th, 2) if th is not None else None,
                           label=mc.plate_label(pl) if pl else None,
                           cli=mc.plate_cli(pl) if pl else None))


def match(theory_list, lo=None, hi=None):
    pairs = mc.match_modes(_peaks(lo, hi), _js(theory_list))
    errs = np.array([q[3] for q in pairs])
    return json.dumps(dict(
        pairs=[dict(peak=q[0], idx=q[1] + 1, analytical=q[2], diff=q[3]) for q in pairs],
        n_peaks=len(_peaks(lo, hi)), n_all=len(S.modes),
        mean=float(errs.mean()) if len(errs) else None,
        rms=float(np.sqrt(np.mean(errs**2))) if len(errs) else None))


def optimize(plate_js, free, lo=None, hi=None):
    pl = _plate(plate_js)
    free = _js(free)
    meas = _peaks(lo, hi)
    if len(meas) < 2:
        return json.dumps(dict(ok=False, msg="Accept at least two modes in the band first"))
    if not free:
        return json.dumps(dict(ok=False, msg="Tick at least one parameter to vary"))
    q, before, after = mc.optimize_plate(pl, meas, mc.PLATE_SPECS, list(free), S.fmax)
    spec = {sp[0]: sp for sp in mc.PLATE_SPECS}
    at_limit = [k for k in free
                if np.isclose(spec[k][4](q[k]), spec[k][2]) or np.isclose(spec[k][4](q[k]), spec[k][3])]
    return json.dumps(dict(ok=True, plate=q, before=before, after=after, at_limit=at_limit,
                           n_peaks=len(meas), cli=mc.plate_cli(q)))


def plate_to_sliders(plate_js):
    pl = _plate(plate_js)
    return json.dumps({sp[0]: sp[4](pl[sp[0]]) for sp in mc.PLATE_SPECS})


def sliders_to_plate(values_js):
    vals = json.loads(values_js)
    pl = dict(mc.DEFAULT_PLATE)
    for sp in mc.PLATE_SPECS:
        if sp[0] in vals:
            pl[sp[0]] = sp[5](float(vals[sp[0]]))
    return json.dumps(pl)


# ----------------------------------------------------------------------------
def exports():
    return json.dumps(dict(modes_csv=mc.table_csv(mc.modes_table(S)),
                           shapes_csv=mc.table_csv(mc.shapes_table(S)),
                           replay=mc.replay_string(S), name=S.ds["name"]))

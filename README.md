# Incremental Circle Fit

Interactive modal analysis of ObieApp roving-point FRF measurements. You pick resonance peaks one at a time. Each peak is circle-fitted to get its natural frequency, loss factor and mode shape, then subtracted from every FRF before you move on to the next one.

There are two ways to use it, with the same fitting code:

- **Web app** (`index.html`): everything in one browser window, nothing to install.
- **Desktop script** (`circle_fit_modes.py`): matplotlib windows, run from the command line.

## Web app

Open `index.html` from a web server (GitHub Pages, or locally as below) in Chrome, Edge or another current browser. Click **📁 Open test folder** and pick an ObieApp test folder (the one containing `TRF/` and `template.json`), or a folder of several tests and choose one. Everything runs in your browser: Python, NumPy and SciPy load through [Pyodide](https://pyodide.org) (15–30 s the first time, cached after that), and your data files are read locally and never uploaded.

The page has the FRF plot on top (click a peak to fit it), the accepted modes on the left, and below: the circle for any point (slider or ↑ ↓), each point's own natural-frequency estimate, a zoom on the fitted peak and the mode shape. **Analytical ▸** opens the plate panel with the sliders, frequency band, Optimize button and pairing table. **Export** downloads `modes.csv` and `mode_shapes.csv` (the same files the desktop script writes) and gives a replay string. **Options** sets the frequency range, zero padding (seconds of zeros added after each raw recording), sign flip and global refinement. **Hide measured** (`m`) shows the residual FRF without the original underneath. Picking a point (on the mode shape or estimates plot, with the slider or ↑ ↓) also switches the top plot to that point's own FRF, residual and fitted model; **FRF: all points / point N** (`p`) switches between that and the rms over all points. Fitting always uses every point. If any raw recordings reach full scale (clipped), the status bar says so. The keyboard shortcuts match the desktop script; **Help** lists them. Accepted modes are remembered in the browser for each test, and offered for restoring the next time you open it.

To run it locally:

```
python3 -m http.server 8000
```

then open http://localhost:8000. (Opening `index.html` directly as a file does not work, because the page fetches `modal_core.py`.)

Files: `index.html`, `web/app.js` (page and plots, using Plotly), `web/app.css` (styling, following ObieApp's theme), `web/worker.js` (runs Python in a background worker) and `web/api.py` (the bridge to `modal_core.py`).

## Code layout

All the maths lives in `modal_core.py`: file loading, circle fitting, the residual bookkeeping, the analytical plate model, peak pairing, optimization and the result tables. Both the desktop script and the web app use it, so they give identical results.

## Requirements (desktop script)

Python 3 with `numpy`, `scipy` and `matplotlib`:

```
pip install -r requirements.txt
```

Tested with Python 3.14, numpy 2.5, scipy 1.18 and matplotlib 3.11.

## Usage (desktop script)

Point the script at an ObieApp test folder (one containing a `TRF/` subfolder of `.trf` files, and optionally `template.json` for point positions):

```
python3 circle_fit_modes.py path/to/Demo/Demo_07
python3 circle_fit_modes.py path/to/Demo/Demo_08 --fmax 2000
```

| Option | Meaning |
|---|---|
| `--fmin`, `--fmax` | Frequency range shown and used for the residual (default 20–7000 Hz) |
| `--modes 34.3,73.6,196:4` | Pre-fit these peaks (Hz, with an optional `:half-width` in bins) before the window opens |
| `--no-gui` | Fit `--modes`, save, and exit |
| `--no-refine` | Use the per-point circle estimates only (skip the global least-squares refinement) |
| `--out DIR` | Output folder (default `CircleFit/<test name>` next to the script) |
| `--plate L,W,t,c` | Analytical cantilever plate to compare against, in mm: length (perpendicular to the clamped edge), width, thickness, clamp depth. Default `150,150,0.86,15` (measured thickness 0.86 mm); `none` turns it off |
| `--pad-seconds S` | Zero padding: add S seconds of zeros after each raw hit recording (`raw/*.wav`, after the template's time cutoff) and rebuild the FRFs, for finer frequency bins. 2 s + 6 s gives 4× more points per peak. It interpolates what was recorded rather than adding information. Needs the raw recordings; 0 (default) uses the `.trf` files |
| `--flip-sign` | Multiply all FRFs by −1. Use it when the accelerometer is mounted upside down relative to the hammer direction, so that circles at points moving in phase with the accelerometer hang down. Frequencies, damping and mode shapes are unchanged |
| `--clamp-k K` | Clamp rotational stiffness for the analytical plate, N·m/rad per metre of clamped edge, or `rigid` (default) |
| `--E`, `--nu`, `--rho` | Plate material (default aluminium: 69 GPa, 0.33, 2700 kg/m³) |

The `peak_Hz` and `half_width_bins` columns in a saved `modes.csv` use the `--modes` format, so you can replay a session.

### Controls

| Key / action | Does |
|---|---|
| click a peak | Fit that peak (snaps to the nearest local maximum of the residual, so small peaks next to big ones can be picked) |
| shift+click | Fit centred exactly on the clicked bin, without snapping |
| `n` | Fit the next peak above the last accepted mode |
| `a` / Enter | Accept: subtract the mode from all FRFs |
| `x` / Esc | Discard the proposed fit |
| `[` / `]` | Narrow / widen the fit band by one bin each side and refit |
| `,` / `.` | Move the fit centre one bin lower / higher and refit |
| `u` | Undo the last accepted mode |
| ← / → or Prev / Next | Step through accepted modes; **Latest** returns to the end |
| `f` | Reopen the fitted-model window if you closed it |
| `t` | Open the analytical tuning window |
| **Point** slider, ↑ / ↓, or click a dot on the mode shape | Show that point's circle in the Nyquist panel (clicking a point in the estimates panel also works) (**Strongest** goes back to the strongest point) |
| scroll wheel | Zoom any plot around the cursor (hold shift for frequency/x only, ctrl for y only) |
| `o` / `p` | Toolbar box zoom / pan, on any plot |
| `0` | Reset the zoom of the plot under the mouse; with the mouse outside the plots, reset all of them |
| `s` / `q` | Save / save and quit |

Each panel remembers its zoom while it shows the same mode (the circle plot remembers it per point), so moving the slider or stepping through modes and back keeps the view you set. Closing the window without `s` or `q` discards the session. Clicks are ignored while the toolbar's zoom or pan is active.

## Reading the window

- **Top:** rms |H| over all points. Grey is measured, blue is the residual after the accepted modes, orange is the sum of accepted modes. A peak still in blue is a mode not yet fitted.
- **Nyquist plot:** receptance H/(−ω²) for the point you picked (by default the strongest). The title gives the point's |A| relative to the largest, how far its data sits from the fitted circle (as a share of the circle's diameter), and how far its phase is from cleanly in or out of phase. Blue dots are the measured bins in the fit band, orange is the fitted circle, ★ is the resonance (f_r), + is the offset from all other modes, and ○ with grey axis lines marks the origin (0,0).
- **Per-point f_r estimates:** each point's own circle-fit natural frequency, as Hz from the final value, against the point number. Filled blue dots were used in the circle-only estimate, a weighted median with marker size showing the weight. Hollow grey dots were excluded because their circle was smaller than 20% of the largest, and red crosses because their data sat more than 30% of a radius off their own circle. The dashed orange line is the circle-only estimate, and the solid line is the final value from the global least-squares fit, which uses every point. Points off the scale are drawn on the edge.
- **Zoom:** the residual before (blue dots = measured bins) and after (green) removing this mode, the mode alone drawn smoothly on a fine grid (orange, ★ at f_r), and the fit band (shaded).
- **Mode shape:** the real part of the mode shape on the measurement grid, scaled to ±1. The black line is the nodal line, and a black square marks the point in the left circle panel. Coloured rings highlight points worth a look, listed under the plot: **green** for the two strongest points, **red** for points (with at least 25% of the largest |A|) whose data sits more than 10% of a diameter off the fitted circle, and **purple** for such points whose phase is more than 20° from cleanly in or out of phase. Red often means a neighbouring mode, noise or a bad tap; purple often means a neighbouring mode or a complex (non-real) mode.

A second window, **fitted model FRF**, shows the measured rms |H| with a dot at every bin next to the accepted modes drawn on a grid 8× finer than the measurement. Each mode is marked and labelled with its natural frequency, so you can see the true peak positions even when they fall between measured bins. A proposed mode shows as a dashed green curve. The window follows the step browser.

Purple dashed lines in the main window, the fitted-model window and the zoom panel are the analytical natural frequencies of the plate given by `--plate`, numbered in order.

A good fit has data points on the circle, a green curve that drops to the local background with no leftover peak, and a smooth mode shape.

## Output

Saved to `CircleFit/<test name>/`:

- `modes.csv`: f_r, η, ζ, Q, the circle-only estimates, fit band and local improvement for each mode
- `mode_shapes.csv`: per point, the real mode shape plus |A| and phase for each mode
- `model_frf.png`: the fitted-model window, with all accepted modes
- `summary.png`: measured data, mode sum and residual, plus the residual after each step
- `mode_shapes.png`: all mode shapes in one grid
- `steps/mode_NN_<f>Hz.png`: the circle-fit figure for each accepted mode

## Method

For each point, a circle is fitted to the receptance Nyquist data in the fit band. The angle-versus-frequency relation on that circle, φ = φ₀ ± 2·atan((1 − ω²/ω_r²)/η), gives f_r and the loss factor η. The per-point estimates are combined using a median weighted by circle size. f_r and η are then refined by one least-squares fit of

  A_p / (ω_r² − ω² + iηω_r²) + B_p

over all points at once. B_p is the offset caused by other modes and is not subtracted. The mode's contribution, −ω²·A_p / (ω_r² − ω² + iηω_r²), is removed from every FRF at all frequencies.

The fit band defaults to about twice the peak's half-power half-width, limited to 2–40 bins each side, and is narrowed so it never reaches a point higher than the peak being fitted (a bigger neighbour).

## Analytical plate frequencies

Press `t` to open the **analytical tuning** window. It has sliders for thickness, length, width, clamp depth, Young's modulus, density, Poisson's ratio and clamp stiffness. The plot shows your accepted modes (red; detected peaks if none are accepted) against the analytical frequencies (purple dashed), with an arrow from each peak to its nearest analytical mode. The table lists each pairing and its percent difference, and the title gives the mean and rms difference, so you can tune the sliders until the analytical model best matches your measurements. The lines follow the sliders as you drag; the other windows update when you release the mouse. To score and optimize over part of the spectrum only, drag across the plot or type a band such as `150-700` in the **band (Hz)** box; peaks outside the band are faded and ignored, and **All frequencies** (or typing `all`) goes back to every peak. **Optimize** adjusts the parameters ticked under "vary when optimizing" (by default E and clamp stiffness) to minimise the rms difference, searching within the slider ranges, then moves the sliders there. Thickness, E and density only affect the frequencies in the combination h·√(E/ρ), so vary at most one of them at a time, or the split between them is arbitrary. Peaks are paired with analytical modes one-to-one (each analytical mode used at most once), so the optimizer can't cheat by piling modes up near the peaks. **Print settings** prints the matching command-line options so you can reuse a setting, and **Reset** returns to the starting values. Pairing is by frequency only, not mode shape, so check that the pairings make physical sense.

The clamp stiffness models an imperfect clamp: the edge is held in place but can rotate against a rotational spring of k N·m/rad per metre of edge. The slider runs from 1 to 10⁶ on a log scale, with the right end being a rigid clamp. Softening the clamp mainly lowers the spanwise bending and torsion modes.

The plate is modelled as a thin (Kirchhoff) rectangular plate, clamped along one full edge and free on the other three. The clamped strip is treated as rigid, so the free span is length − clamp depth. With a rigid clamp, natural frequencies come from a Rayleigh–Ritz solution with 24 × 24 Legendre-polynomial trial functions, which reproduces Leissa's published values for a square cantilever plate (λ = 3.471, 8.507, 21.28, 27.20, 30.96) and is converged for the first ~50 modes. The frequencies are printed at startup.

## Limitations

- **Frequency resolution:** the bins are evenly spaced (about 0.5 Hz in the demo data), and a peak's half-power width is about η·f_r. Lightly damped low-frequency modes therefore span only one or two bins, and their circle-only η is unreliable. Use the refined values for those modes. A longer acquisition record (Δf = 1/T) is the real fix.
- **Close modes:** this is a single-mode method. Overlapping peaks are better handled by a method that fits several modes at once, such as RFP or PolyMAX.

## ObieApp `.trf` format

The reader follows ObieApp's own `trf_fileio.py`: a 110-byte little-endian header (`<I4d2s3d11f4s`) holding the frequency step, start and end frequency, `fComplex` and the number of bins, then the bins, then an optional `\0OBIE_META` text block.

| `fComplex` | Data per bin | Supported |
|---|---|---|
| 0 | magnitude only | no (circle fitting needs complex FRFs) |
| 1 | real, imaginary | yes |
| 2 | real, imaginary, coherence | yes (what Acquire writes) |

Bins must start at 0 Hz.

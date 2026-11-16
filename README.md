# Incremental Circle Fit

Interactive modal analysis of ObieApp roving-point FRF measurements. You pick resonance peaks one at a time. Each peak is circle-fitted to get its natural frequency, loss factor and mode shape, then subtracted from every FRF before you move on to the next one.

## Requirements

Python 3 with `numpy`, `scipy` and `matplotlib`:

```
pip install numpy scipy matplotlib
```

## Usage

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

The `peak_Hz` and `half_width_bins` columns in a saved `modes.csv` use the `--modes` format, so you can replay a session.

### Controls

| Key / action | Does |
|---|---|
| click a peak | Fit that peak (snaps to the nearest residual maximum) |
| `n` | Fit the next peak above the last accepted mode |
| `a` / Enter | Accept: subtract the mode from all FRFs |
| `x` / Esc | Discard the proposed fit |
| `[` / `]` | Narrow / widen the fit band by one bin each side and refit |
| `u` | Undo the last accepted mode |
| ← / → or Prev / Next | Step through accepted modes; **Latest** returns to the end |
| `s` / `q` | Save / save and quit |

Closing the window without `s` or `q` discards the session. Clicks are ignored while the toolbar's zoom or pan is active.

## Reading the window

- **Top:** rms |H| over all points. Grey is measured, blue is the residual after the accepted modes, orange is the sum of accepted modes. A peak still in blue is a mode not yet fitted.
- **Nyquist plots:** receptance H/(−ω²) for the two points that respond most strongly. Blue dots are the measured bins in the fit band, orange is the fitted circle, ★ is the resonance (f_r), and + is the offset from all other modes.
- **Zoom:** the residual before (blue) and after (green) removing this mode, the mode alone (orange), and the fit band (shaded).
- **Mode shape:** the real part of the mode shape on the measurement grid, scaled to ±1. The black line is the nodal line.

A good fit has data points on the circle, a green curve that drops to the local background with no leftover peak, and a smooth mode shape.

## Output

Saved to `CircleFit/<test name>/`:

- `modes.csv`: f_r, η, ζ, Q, the circle-only estimates, fit band and local improvement for each mode
- `mode_shapes.csv`: per point, the real mode shape plus |A| and phase for each mode
- `summary.png`: measured data, mode sum and residual, plus the residual after each step
- `mode_shapes.png`: all mode shapes in one grid
- `steps/mode_NN_<f>Hz.png`: the circle-fit figure for each accepted mode

## Method

For each point, a circle is fitted to the receptance Nyquist data in the fit band. The angle-versus-frequency relation on that circle, φ = φ₀ ± 2·atan((1 − ω²/ω_r²)/η), gives f_r and the loss factor η. The per-point estimates are combined using a median weighted by circle size. f_r and η are then refined by one least-squares fit of

  A_p / (ω_r² − ω² + iηω_r²) + B_p

over all points at once. B_p is the offset caused by other modes and is not subtracted. The mode's contribution, −ω²·A_p / (ω_r² − ω² + iηω_r²), is removed from every FRF at all frequencies.

The fit band defaults to about twice the peak's half-power half-width, limited to 2–40 bins each side.

## Limitations

- **Frequency resolution:** the bins are evenly spaced (about 0.5 Hz in the demo data), and a peak's half-power width is about η·f_r. Lightly damped low-frequency modes therefore span only one or two bins, and their circle-only η is unreliable. Use the refined values for those modes. A longer acquisition record (Δf = 1/T) is the real fix.
- **Close modes:** this is a single-mode method. Overlapping peaks are better handled by a method that fits several modes at once, such as RFP or PolyMAX.

## ObieApp `.trf` format

The script reads this layout, which was worked out from the files themselves (little-endian):

| Offset | Type | Field |
|---|---|---|
| 0x26 | float64 | Δf (Hz) |
| 0x36 | float64 | f_max (Hz) |
| 0x42 | float32 | number of frequency bins N |
| 110 | N × (float64, float64, float64) | real, imaginary, coherence |

The bin data is followed by a `\0OBIE_META\n…` text trailer.

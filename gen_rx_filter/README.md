# Rx Filter Design

## Overview

The Rx filter chain consists of two cascaded filters:

1. **AAF (Anti-Aliasing Filter)** — equiripple lowpass FIR designed via the Remez exchange algorithm (`FilterDesign`). Attenuates out-of-band energy before decimation. Passband edge and stopband edge are configured per sample-rate mode.

2. **Equalizer (EQZ) approximation** — least-squares FIR designed via `firls` (`FirlsFilterDesign`). Approximates the target EQZ frequency response (magnitude only) using a windowed least-squares fit, with an explicit stopband weight to suppress out-of-band leakage.

## Filter Design Classes

| Class | Algorithm | Phase | Input |
|---|---|---|---|
| `FilterDesign` | Remez (equiripple) | Linear | passband/stopband edges, ripple/attenuation specs |
| `FirlsFilterDesign` | Weighted least-squares (`firls`) | Linear | complex frequency response (magnitude taken) |

## Compared Responses

Three responses are evaluated and plotted:

- **equalizer** — raw target EQZ response (ideal reference)
- **comb. of AAF and equalizer** — cascade of the Remez AAF with the raw EQZ target
- **least square (windowed)** — `FirlsFilterDesign` fit to the EQZ magnitude with Hamming window

## Results

### Magnitude Response
![](figure/aaf_resp/magnitude_resp.png)

### In-Band Magnitude Error (relative to EQZ target)
![](figure/aaf_resp/inband_error.png)

## How to Run

```bash
python3 gen_rx_filter/gen_aaf_coef.py
```

Output figures are saved to `gen_rx_filter/figure/aaf_resp/`.

---

# Complex FIR Filter Design via Convex Optimisation (`design_fir_cvxpy.py`)

## Overview

`design_fir_cvxpy.py` designs a **complex-valued, non-symmetric FIR filter** h[n] using
[CVXPY](https://www.cvxpy.org/). The frequency response is formulated as a linear map over
a sampled frequency grid and the design is cast as a second-order cone program (SOCP):

```
H(w) = sum_{n=0}^{N-1} h[n] * exp(-j * w * n)  =  A @ h

A[k, n] = exp(-j * w[k] * n)      (K × N evaluation matrix)
```

## Optimisation Modes

### `minimax` (default)
Minimises the worst-case deviation δ jointly across the passband and stopband:

```
minimise   δ
subject to |A_pb @ h - Hd_pb| ≤ δ    ∀ w ∈ passband
           |A_sb @ h|          ≤ δ    ∀ w ∈ stopband
```

### `fixed`
Feasibility / minimum-error problem with separate per-band tolerances:

```
minimise   Σ |A_pb @ h - Hd_pb|
subject to |A_pb @ h - Hd_pb| ≤ delta_p    ∀ w ∈ passband
           |A_sb @ h|          ≤ delta_s    ∀ w ∈ stopband
```

Both modes use CVXPY's native complex-variable support (`cp.Variable(N, complex=True)`);
`cp.abs()` on a complex expression is automatically recognised as a convex SOC constraint.

## Installation

```bash
# cvxpy and its solvers (osqp is skipped due to a pip/toml parser issue on this system)
python3.8 -m pip install cvxpy ecos scs clarabel --user
```

## API

| Function | Description |
|---|---|
| `build_frequency_matrix(freqs, N)` | Returns A[k,n] = exp(-j·w[k]·n), shape (K, N) |
| `design_complex_fir_cvxpy(...)` | Runs the CVXPY SOCP; returns complex h of shape (N,) |
| `evaluate_response(h, freqs)` | Evaluates H(w) = A @ h at arbitrary angular frequencies |
| `plot_response(...)` | Saves magnitude (dB) and phase PNG using `scipy.signal.freqz` |
| `save_coefficients(...)` | Writes h to `.npy` and `.csv` (columns: real, imag) |

## Command-Line Usage

```
usage: design_fir_cvxpy.py [-h] [--N N] [--fs FS]
                           [--pb-low PB_LOW] [--pb-high PB_HIGH]
                           [--sb-low SB_LOW] [--sb-high SB_HIGH]
                           [--sb-low2 SB_LOW2] [--sb-high2 SB_HIGH2]
                           [--delta-p DELTA_P] [--delta-s DELTA_S]
                           [--mode {minimax,fixed}]
                           [--n-grid N_GRID] [--solver SOLVER]
                           [--out-dir OUT_DIR] [--verbose]
```

All frequency edges are **normalised** (0–0.5, where 0.5 = Nyquist).

| Argument | Default | Description |
|---|---|---|
| `--N` | 31 | Number of filter taps |
| `--fs` | 1.0 | Sample rate in Hz (axis labels only) |
| `--pb-low` | 0.10 | Passband lower edge |
| `--pb-high` | 0.20 | Passband upper edge |
| `--sb-low` | 0.00 | First stopband lower edge |
| `--sb-high` | 0.07 | First stopband upper edge |
| `--sb-low2` | 0.26 | Second stopband lower edge |
| `--sb-high2` | 0.50 | Second stopband upper edge |
| `--delta-p` | 0.05 | Passband ripple tolerance (fixed mode) |
| `--delta-s` | 0.01 | Stopband attenuation tolerance (fixed mode) |
| `--mode` | minimax | `minimax` or `fixed` |
| `--n-grid` | 256 | Frequency grid points per band |
| `--solver` | auto | CVXPY solver: `SCS`, `CLARABEL`, … |
| `--out-dir` | `cvxpy_fir_output` | Output directory |
| `--verbose` | off | Show solver output |

## Examples

### Minimal bandpass filter (minimax, default settings)

```bash
python3.8 gen_rx_filter/design_fir_cvxpy.py
```

Designs a 31-tap complex bandpass for the band [0.10, 0.20]·fs with
stopbands [0.00, 0.07] and [0.26, 0.50]·fs.

### Specify tap count and solver

```bash
python3.8 gen_rx_filter/design_fir_cvxpy.py --N 63 --solver SCS
```

### Fixed-tolerance mode with custom band edges

```bash
python3.8 gen_rx_filter/design_fir_cvxpy.py \
    --N 63 --mode fixed \
    --pb-low 0.15 --pb-high 0.30 \
    --sb-low 0.00 --sb-high 0.10 \
    --sb-low2 0.35 --sb-high2 0.50 \
    --delta-p 0.10 --delta-s 0.05 \
    --out-dir output/bandpass_fixed
```

### Use in a script

```python
import numpy as np
from gen_rx_filter.design_fir_cvxpy import (
    build_frequency_matrix,
    design_complex_fir_cvxpy,
    evaluate_response,
    plot_response,
    save_coefficients,
)

# Frequency grids (angular: normalised × 2π)
pb_freqs = np.linspace(0.10, 0.20, 256) * 2 * np.pi
sb_freqs = np.concatenate([
    np.linspace(0.00, 0.07, 256) * 2 * np.pi,
    np.linspace(0.26, 0.50, 256) * 2 * np.pi,
])

# Desired response: linear phase (pure delay of (N-1)/2 samples)
N = 31
Hd_pb = np.exp(-1j * pb_freqs * (N - 1) / 2)

# Design
h = design_complex_fir_cvxpy(
    N=N,
    passband_freqs=pb_freqs,
    stopband_freqs=sb_freqs,
    Hd_passband=Hd_pb,
    mode='minimax',
    solver='SCS',
)

# Evaluate, plot, save
H_pb = evaluate_response(h, pb_freqs)
print(f"Peak passband error: {np.max(np.abs(H_pb - Hd_pb)):.4f}")

save_coefficients(h, out_dir='output', name='my_filter')
plot_response(h, fs=1.0, out_dir='output', title='My Complex Bandpass')
```

## Outputs

All files are written to `--out-dir` (default: `cvxpy_fir_output/`):

| File | Content |
|---|---|
| `complex_bandpass.npy` | Complex tap vector h, shape (N,) |
| `complex_bandpass.csv` | Two-column text: `real,imag` per tap |
| `filter_response.png` | Magnitude (dB) and phase plots |

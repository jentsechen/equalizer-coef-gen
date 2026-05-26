# gen_rx_filter

Run all commands from the **repository root**.

---

## Design scripts

### CVX lowpass (CVXPY SOCP)

```bash
python3 gen_rx_filter/run_eqz_lowpass.py
python3 gen_rx_filter/run_eqz_lowpass.py --N 95 --mode minimax
python3 gen_rx_filter/run_eqz_lowpass.py --no-transition-band   # disable TB constraint
```

Output → `gen_rx_filter/figure/eqz_lowpass/`

### Remez lowpass (complex Lawson IRLS)

```bash
python3 gen_rx_filter/run_eqz_lowpass_remez.py
python3 gen_rx_filter/run_eqz_lowpass_remez.py --N 95 --n-iter 200
```

Output → `gen_rx_filter/figure/eqz_lowpass_remez/`

---

## Comparison figures

### CVX vs Remez overlay

```bash
python3 gen_rx_filter/compare_lowpass_methods.py
python3 gen_rx_filter/compare_lowpass_methods.py --N 95
```

Output → `gen_rx_filter/figure/compare_lowpass_methods/`

---

## Common options

| Option      | Default | Description                     |
| ----------- | ------- | ------------------------------- |
| `--N`       | 63      | Number of filter taps           |
| `--f-pass`  | 162.5   | Passband half-bandwidth (MHz)   |
| `--f-stop`  | 250.0   | Stopband edge (MHz)             |
| `--delta-p` | 0.05    | Passband ripple tolerance       |
| `--delta-s` | 0.10    | Stopband attenuation tolerance  |
| `--n-grid`  | 512     | Frequency grid points per band  |
| `--verbose` | off     | Print per-iteration diagnostics |

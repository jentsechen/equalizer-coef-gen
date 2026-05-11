"""
Design a complex lowpass FIR filter whose passband target is the equalizer
frequency response from gen_eqz_freq_resp().

Signal chain context
--------------------
  fs = 750 MHz  (two-sided, so Nyquist = ±375 MHz)
  Passband  : -162.5 MHz to +162.5 MHz  (half-bandwidth = 162.5 MHz)
  Stopband  : |f| > f_stop_mhz          (default 195 MHz, matching gen_aaf_coef.py)
  Desired   : Hd(f) = eqz_resp at each passband frequency,  0 in stopband

The desired response is obtained by:
  1. Calling gen_eqz_freq_resp(sig=DistortedSig.UCDC, n_taps=4)
     → 256-bin complex FFT at 750 MHz, bin spacing ≈ 2.93 MHz
  2. Using interp_fft_response() to interpolate it onto the dense CVXPY
     frequency grid (passband and stopband separately).

Run from the repository root:
    python3.8 gen_rx_filter/run_eqz_lowpass.py
    python3.8 gen_rx_filter/run_eqz_lowpass.py --N 63 --mode fixed
"""

import argparse
import os
import sys

import numpy as np

# Ensure both the project root and gen_rx_filter directory are on the path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from gen_rx_filter import gen_eqz_freq_resp, DistortedSig
from design_fir_cvxpy import (
    design_complex_fir_cvxpy,
    evaluate_response,
    interp_fft_response,
    plot_response,
    save_coefficients,
)


# ---------------------------------------------------------------------------
# Main design routine
# ---------------------------------------------------------------------------

def design_eqz_lowpass(
    N: int = 127,
    fs_mhz: float = 750.0,
    f_pass_mhz: float = 162.5,
    f_stop_mhz: float = 250.0,
    mode: str = 'fixed',
    delta_p: float = 0.02,
    delta_s: float = 0.05,
    lambda_reg: float = 1e-3,
    n_grid: int = 512,
    solver: str = None,
    out_dir: str = 'gen_rx_filter/figure/eqz_lowpass',
    verbose: bool = False,
) -> np.ndarray:
    """Design a complex lowpass FIR matched to the UCDC equalizer response.

    Args:
        N: Number of filter taps.
        fs_mhz: Sample rate in MHz.
        f_pass_mhz: Passband half-bandwidth in MHz (±f_pass_mhz).
        f_stop_mhz: Stopband start frequency in MHz.
        mode: 'minimax' or 'fixed'.
        n_grid: Number of frequency grid points per band.
        solver: CVXPY solver name (None → auto).
        out_dir: Output directory for saved files.
        verbose: Show solver output.

    Returns:
        Designed complex tap vector h of shape (N,).
    """
    # ------------------------------------------------------------------ #
    # Step 1: get the 256-bin desired frequency response (eqz_resp)       #
    # ------------------------------------------------------------------ #
    eqz_resp = gen_eqz_freq_resp(sig=DistortedSig.UCDC, n_taps=4)
    # eqz_resp[k] corresponds to physical frequency  k * fs_mhz/256  MHz
    # (standard two-sided FFT order: DC, positive, then negative bins)

    # ------------------------------------------------------------------ #
    # Step 2: build frequency grids (angular, rad/sample)                 #
    #   normalised freq = physical_MHz / fs_mhz  (0 → DC, 0.5 → Nyquist) #
    #   angular freq    = normalised * 2π                                  #
    # ------------------------------------------------------------------ #
    pb_norm  = f_pass_mhz / fs_mhz          # e.g. 162.5/750 = 0.2167
    sb_norm  = f_stop_mhz / fs_mhz          # e.g. 195.0/750 = 0.2600

    # Passband: two-sided lowpass → positive AND negative half
    pb_freqs_pos = np.linspace(0.0,     pb_norm, n_grid) * 2 * np.pi
    pb_freqs_neg = np.linspace(-pb_norm, 0.0,    n_grid) * 2 * np.pi
    pb_freqs = np.concatenate([pb_freqs_neg, pb_freqs_pos])

    # Stopband: both sides beyond f_stop_mhz
    sb_freqs_pos = np.linspace(sb_norm, 0.5,     n_grid) * 2 * np.pi
    sb_freqs_neg = np.linspace(-0.5,   -sb_norm, n_grid) * 2 * np.pi
    sb_freqs = np.concatenate([sb_freqs_neg, sb_freqs_pos])

    # Transition band: both sides between passband and stopband edges.
    # Start one grid step inside the passband edge to avoid overlapping the
    # passband constraint at exactly pb_norm.
    tb_step = (sb_norm - pb_norm) / (n_grid // 2)
    tb_freqs_pos = np.linspace(pb_norm + tb_step, sb_norm, n_grid // 2) * 2 * np.pi
    tb_freqs_neg = np.linspace(-sb_norm, -pb_norm - tb_step, n_grid // 2) * 2 * np.pi
    tb_freqs = np.concatenate([tb_freqs_neg, tb_freqs_pos])

    # ------------------------------------------------------------------ #
    # Step 3: interpolate eqz_resp onto the passband grid                 #
    # ------------------------------------------------------------------ #
    # interp_fft_response maps 256-bin FFT → complex values at target_freqs_rad
    Hd_pb = interp_fft_response(eqz_resp, fs=fs_mhz, target_freqs_rad=pb_freqs)

    # Upper bound for transition band: peak passband gain + passband tolerance
    # + small margin, so the constraint is always strictly compatible with the
    # passband constraint regardless of where |Hd| peaks.
    delta_t = float(np.max(np.abs(Hd_pb))) + delta_p + 0.02

    # ------------------------------------------------------------------ #
    # Step 4: solve the CVXPY SOCP                                        #
    # ------------------------------------------------------------------ #
    print(f"Designing {N}-tap complex lowpass FIR  [mode={mode}]")
    print(f"  fs         = {fs_mhz} MHz")
    print(f"  Passband   = ±{f_pass_mhz} MHz  (±{pb_norm:.4f} × fs)")
    print(f"  Stopband   ≥  {f_stop_mhz} MHz  ( {sb_norm:.4f} × fs)")
    print(f"  Desired    = eqz_resp (UCDC, n_taps=4),  256-bin FFT")
    if mode == 'fixed':
        print(f"  delta_p    = {delta_p}  (passband),  delta_s = {delta_s}  (stopband)")
    print(f"  delta_t    = {delta_t:.3f}  (transition band upper bound)")
    print(f"  lambda_reg = {lambda_reg}  (tap-energy regularisation)")

    h = design_complex_fir_cvxpy(
        N=N,
        passband_freqs=pb_freqs,
        stopband_freqs=sb_freqs,
        Hd_passband=Hd_pb,
        mode=mode,
        delta_p=delta_p,
        delta_s=delta_s,
        lambda_reg=lambda_reg,
        transition_freqs=tb_freqs,
        delta_t=delta_t,
        solver=solver,
        verbose=verbose,
    )

    # ------------------------------------------------------------------ #
    # Step 5: report achieved performance                                 #
    # ------------------------------------------------------------------ #
    H_pb  = evaluate_response(h, pb_freqs)
    H_sb  = evaluate_response(h, sb_freqs)
    pb_err = np.max(np.abs(H_pb - Hd_pb))
    sb_pk  = np.max(np.abs(H_sb))
    print(f"\n  Passband peak |H - Hd| : {pb_err:.5f}")
    print(f"  Stopband peak |H|      : {sb_pk:.5f}  ({20*np.log10(sb_pk+1e-12):.1f} dB)")

    # ------------------------------------------------------------------ #
    # Step 6: save and plot                                               #
    # ------------------------------------------------------------------ #
    save_coefficients(h, out_dir=out_dir, name='eqz_lowpass')
    plot_response(
        h, fs=fs_mhz, out_dir=out_dir,
        title=f'EQZ Lowpass FIR  N={N}  ±{f_pass_mhz} MHz  mode={mode}',
        passband_edges=(0.0, pb_norm),
        freq_unit='MHz',
        Hd_freqs_rad=pb_freqs,
        Hd=Hd_pb,
    )
    return h


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Design a complex lowpass FIR matched to the UCDC equalizer response.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--N',          type=int,   default=127,
                   help='Number of filter taps')
    p.add_argument('--fs',         type=float, default=750.0,
                   help='Sample rate in MHz')
    p.add_argument('--f-pass',     type=float, default=162.5,
                   help='Passband half-bandwidth in MHz')
    p.add_argument('--f-stop',     type=float, default=250.0,
                   help='Stopband start frequency in MHz')
    p.add_argument('--mode',       choices=['minimax', 'fixed'], default='fixed',
                   help='Optimisation mode')
    p.add_argument('--delta-p',    type=float, default=0.02,
                   help='Passband error tolerance (fixed mode)')
    p.add_argument('--delta-s',    type=float, default=0.05,
                   help='Stopband attenuation tolerance (fixed mode)')
    p.add_argument('--lambda-reg', type=float, default=1e-3,
                   help='L2 tap-energy regularisation weight; suppresses transition-band spikes')
    p.add_argument('--n-grid',     type=int,   default=512,
                   help='Frequency grid points per band')
    p.add_argument('--solver',     type=str,   default=None,
                   help='CVXPY solver (SCS, CLARABEL, …). Default: auto')
    p.add_argument('--out-dir',    type=str,
                   default=os.path.join(_HERE, 'figure', 'eqz_lowpass'),
                   help='Output directory')
    p.add_argument('--verbose',    action='store_true',
                   help='Show solver output')
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()
    design_eqz_lowpass(
        N=args.N,
        fs_mhz=args.fs,
        f_pass_mhz=args.f_pass,
        f_stop_mhz=args.f_stop,
        mode=args.mode,
        delta_p=args.delta_p,
        delta_s=args.delta_s,
        lambda_reg=args.lambda_reg,
        n_grid=args.n_grid,
        solver=args.solver,
        out_dir=args.out_dir,
        verbose=args.verbose,
    )

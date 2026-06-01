"""
Design a complex lowpass FIR filter whose passband target is the equalizer
frequency response, using the complex Remez algorithm (Lawson IRLS).

This script mirrors run_eqz_lowpass.py but replaces the CVXPY SOCP solver
with a complex Chebyshev approximation algorithm based on Lawson's iteratively
reweighted least-squares (IRLS).  The algorithm converges to the minimax
(equiripple) solution:

    min_h  max_k  W_k |H(ω_k) − Hd_k|

Signal chain context
--------------------
  fs = 750 MHz  (two-sided, Nyquist = ±375 MHz)
  Passband  : −162.5 MHz to +162.5 MHz  (half-bandwidth = 162.5 MHz)
  Stopband  : |f| > f_stop_mhz          (default 250 MHz)
  Desired   : Hd(f) = eqz_resp at passband frequencies, 0 in stopband

Algorithm: Lawson IRLS (complex Remez analog)
--------------------------------------------
The DTFT of an N-tap complex FIR is linear in the tap vector h:

    H(ω) = A(ω) h,    A[k, n] = exp(−j ω_k n)

Lawson's algorithm iteratively solves a weighted least-squares problem whose
weight vector converges to one that concentrates on the worst-case (extremal)
frequency points, giving the Chebyshev (minimax) solution:

  1. Initialise per-band Lawson weights λ_k = 1/K_band.
  2. Form combined weights  w_k = B_k · λ_k  (B encodes band tolerances).
  3. Solve weighted LS:  h ← argmin_h ‖diag(√w)(Ah − Hd)‖².
  4. Compute error       E_k = H(ω_k) − Hd_k.
  5. Update per-band:    λ_k ← λ_k |E_k|,  normalise each band.
  6. Repeat from step 2 until the peak error stops decreasing.

Band weight encoding:
  Passband:        B_k = 1
  Stopband:        B_k = sb_weight  (= delta_p/delta_s in 'fixed' mode)
  Transition band: B_k = sb_weight / delta_t_k  (tapered, optional)

At convergence the weighted peak error is equiripple across bands.

Run from the repository root:
    python3.8 gen_rx_filter/run_eqz_lowpass_remez.py
    python3.8 gen_rx_filter/run_eqz_lowpass_remez.py --N 63 --n-iter 150
"""

import argparse
import os
import sys
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.linalg import lstsq as scipy_lstsq

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "trx_board_meas"))

from gen_rx_filter import gen_eqz_freq_resp, DistortedSig
from s2p import load_freq_resp_as_fft
from gen_aaf_coef import compute_responses as compute_aaf_responses
from design_fir_cvxpy import (
    build_frequency_matrix,
    evaluate_response,
    interp_fft_response,
    plot_response,
    save_coefficients,
)


# ---------------------------------------------------------------------------
# Complex Remez core (Lawson IRLS)
# ---------------------------------------------------------------------------


def design_complex_fir_remez(
    N: int,
    passband_freqs: np.ndarray,
    stopband_freqs: np.ndarray,
    Hd_passband: np.ndarray,
    sb_weight: float = 1.0,
    transition_freqs: Optional[np.ndarray] = None,
    Hd_transition: Optional[np.ndarray] = None,
    delta_t: Union[float, np.ndarray] = 1.0,
    tb_weight: float = 1.0,
    n_iter: int = 150,
    tol: float = 1e-6,
    verbose: bool = False,
) -> np.ndarray:
    """Complex FIR design via Lawson IRLS (complex Remez algorithm).

    Minimises the weighted Chebyshev criterion over the concatenated passband
    and stopband grids:

        δ* = min_h  max_k  W_k |H(ω_k) − Hd_k|

    The band weight W_k = B_k · λ_k consists of a fixed per-band factor B_k
    and an adaptive per-point Lawson weight λ_k.  The Lawson update

        λ_k ← λ_k · |E_k| / (per-band normaliser)

    drives λ to concentrate mass at the extremal (worst-case) frequencies,
    which is the discrete analogue of the equiripple (Chebyshev) condition.

    Args:
        N: Number of filter taps.
        passband_freqs: Angular frequencies (rad/sample) in the passband.
        stopband_freqs: Angular frequencies (rad/sample) in the stopband.
        Hd_passband: Desired complex response at passband_freqs.
        sb_weight: Stopband weight relative to passband.  Setting this to
            delta_p/delta_s makes the equiripple condition match the target
            tolerances so that peak_pb_err ≈ delta_p and peak_sb ≈ delta_s
            at convergence.
        transition_freqs: Optional angular frequencies in the transition band.
        Hd_transition: Desired magnitude response at transition_freqs (real,
            positive).  When provided, the solver tracks this ramp directly
            with weight tb_weight.  When None, falls back to a zero-desired /
            tapered-weight upper-bound encoded via delta_t.
        delta_t: Upper magnitude bound used only when Hd_transition is None.
            Scalar or ndarray of length K_tb; tighter bound → larger weight.
        tb_weight: Transition band weight relative to passband (default 1.0).
            Independent of sb_weight so the transition band constraint is not
            weakened when sb_weight is small (large delta_s).
        n_iter: Maximum Lawson iterations.
        tol: Convergence threshold: stop when the relative change in the
            combined peak error falls below this value.
        verbose: Print per-iteration diagnostics.

    Returns:
        Complex tap vector h of shape (N,).
    """
    K_pb = len(passband_freqs)
    K_sb = len(stopband_freqs)

    # Build unified frequency grid and desired response vector
    freq_parts = [passband_freqs, stopband_freqs]
    Hd_parts = [Hd_passband, np.zeros(K_sb, dtype=complex)]

    K_tb = 0
    if transition_freqs is not None:
        K_tb = len(transition_freqs)
        freq_parts.append(transition_freqs)
        if Hd_transition is not None:
            Hd_parts.append(Hd_transition.astype(complex))
        else:
            Hd_parts.append(np.zeros(K_tb, dtype=complex))

    all_freqs = np.concatenate(freq_parts)  # (K,)
    Hd_all = np.concatenate(Hd_parts)  # (K,)
    K = len(all_freqs)

    # If a transition-band magnitude ramp was provided, store it so we can
    # do a phase-adaptive update each iteration: desired = |ramp| * exp(j*phase(H)).
    # This makes the error purely a magnitude error, avoiding the phase-mismatch
    # blowup that occurs when a real-valued ramp is used as a complex desired.
    Hd_tb_mag = np.abs(Hd_all[K_pb + K_sb :]).copy() if (K_tb > 0 and Hd_transition is not None) else None

    # Fixed band weights B_k ─ encode relative tolerance between bands
    B = np.ones(K)
    B[K_pb : K_pb + K_sb] = sb_weight
    if K_tb > 0:
        if Hd_transition is not None:
            # Desired ramp provided: use tb_weight (independent of sb_weight)
            B[K_pb + K_sb :] = tb_weight
        else:
            dt = np.atleast_1d(delta_t)
            if dt.size == 1:
                dt = np.full(K_tb, float(dt[0]))
            dt = np.maximum(dt, 1e-9)
            B[K_pb + K_sb :] = sb_weight / dt

    # DFT evaluation matrix  A[k, n] = exp(−j ω_k n)
    A = build_frequency_matrix(all_freqs, N)  # (K, N)

    # Initialise Lawson weights: uniform within each band with Σ_band λ = 1.
    # This means w_k = B_k / K_band initially (uniform per point), and the
    # per-band normalisation in the Lawson update keeps Σ_band λ = 1 throughout,
    # so the total weight per band Σ_band w_k = B_band_avg stays fixed.
    lam = np.empty(K)
    lam[:K_pb] = 1.0 / K_pb
    lam[K_pb : K_pb + K_sb] = 1.0 / K_sb
    if K_tb > 0:
        lam[K_pb + K_sb :] = 1.0 / K_tb

    h = None
    prev_peak = np.inf
    eps_floor = 1e-15  # prevents weights collapsing to zero

    for it in range(n_iter):
        # ------------------------------------------------------------------ #
        # Weighted least-squares solve                                        #
        # ------------------------------------------------------------------ #
        w = B * lam  # combined weights (K,)
        W_sqrt = np.sqrt(w)
        Aw = W_sqrt[:, None] * A  # (K, N)  pre-weighted matrix
        Hdw = W_sqrt * Hd_all  # (K,)    pre-weighted desired

        h, _, _, _ = scipy_lstsq(Aw, Hdw)

        # Phase-adaptive transition: align desired phase to current filter so
        # the error reduces to a pure magnitude error |H(ω)| − ramp(ω)|.
        if Hd_tb_mag is not None:
            H_tb = A[K_pb + K_sb :] @ h
            Hd_all[K_pb + K_sb :] = Hd_tb_mag * np.exp(1j * np.angle(H_tb + 1e-12))

        # ------------------------------------------------------------------ #
        # Error diagnostics                                                   #
        # ------------------------------------------------------------------ #
        E = A @ h - Hd_all  # (K,)  complex error at all points
        abs_E = np.abs(E)  # (K,)  |error| per point

        pb_peak = float(np.max(abs_E[:K_pb]))
        sb_peak = float(np.max(abs_E[K_pb : K_pb + K_sb]))
        tb_peak = float(np.max(abs_E[K_pb + K_sb :])) if K_tb > 0 else 0.0

        if verbose:
            msg = f"  iter {it + 1:3d}: pb_peak={pb_peak:.6f}  sb_peak={sb_peak:.6f}"
            if K_tb > 0:
                msg += f"  tb_peak={tb_peak:.6f}"
            print(msg)

        # ------------------------------------------------------------------ #
        # Convergence check                                                   #
        # Equiripple condition: the weighted peak errors across bands should  #
        # be equal at optimum, i.e.  pb_peak ≈ sb_weight * sb_peak.         #
        # Measure the spread (max − min) of weighted band peaks.             #
        # ------------------------------------------------------------------ #
        weighted_peaks = [pb_peak, sb_weight * sb_peak]
        current_peak = max(weighted_peaks)
        band_spread = max(weighted_peaks) - min(weighted_peaks)
        rel_spread = band_spread / (current_peak + 1e-15)

        rel_change = abs(current_peak - prev_peak) / (current_peak + 1e-15)
        if rel_change < tol and rel_spread < tol * 10:
            if verbose:
                print(
                    f"  Converged at iteration {it + 1} "
                    f"(Δpeak/peak={rel_change:.2e}, spread={rel_spread:.2e})."
                )
            break
        prev_peak = current_peak

        # ------------------------------------------------------------------ #
        # Lawson weight update – per-band normalised                         #
        # ------------------------------------------------------------------ #
        # λ_k ← λ_k · |E_k|, then normalise within each band so that
        # Σ_band λ_k = 1.  Because the combined weight is w_k = B_k · λ_k,
        # this keeps Σ_band w_k = B_band_avg constant and preserves the
        # intended relative weighting (sb_weight) between bands.

        def _lawson_update(lam_band: np.ndarray, err_band: np.ndarray) -> np.ndarray:
            lam_new = lam_band * (err_band + eps_floor)
            return lam_new / (np.sum(lam_new) + eps_floor)  # normalise sum → 1

        lam_pb = _lawson_update(lam[:K_pb], abs_E[:K_pb])
        lam_sb = _lawson_update(lam[K_pb : K_pb + K_sb], abs_E[K_pb : K_pb + K_sb])

        parts = [lam_pb, lam_sb]
        if K_tb > 0:
            lam_tb = _lawson_update(lam[K_pb + K_sb :], abs_E[K_pb + K_sb :])
            parts.append(lam_tb)

        lam = np.concatenate(parts)

    return h


# ---------------------------------------------------------------------------
# LaTeX parameter export (unchanged from run_eqz_lowpass.py)
# ---------------------------------------------------------------------------


def _write_params_tex(
    path: str,
    N: int,
    fs_mhz: float,
    f_pass_mhz: float,
    f_stop_mhz: float,
    delta_p: float,
    delta_s: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nyquist = fs_mhz / 2.0
    lines = [
        r"\begin{tabular}{ll}",
        r"  \toprule",
        r"  Parameter & Value \\",
        r"  \midrule",
        rf"  Sample rate $f_s$ & {fs_mhz:.0f}\,MHz (two-sided, Nyquist $= \pm{nyquist:.0f}$\,MHz) \\",
        rf"  Passband edge $f_p$ & $\pm{f_pass_mhz:.1f}$\,MHz \\",
        rf"  Stopband edge $f_{{stop}}$ & $\pm{f_stop_mhz:.1f}$\,MHz \\",
        rf"  Filter taps $N$ & {N} \\",
        rf"  Passband tolerance $\delta_p$ & {delta_p} \\",
        rf"  Stopband tolerance $\delta_s$ & {delta_s} \\",
        r"  \bottomrule",
        r"\end{tabular}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved params → {path}")


# ---------------------------------------------------------------------------
# Helpers for design_eqz_lowpass_remez
# ---------------------------------------------------------------------------


def _build_frequency_grids(
    pb_norm: float,
    sb_norm: float,
    tb_start_norm: float,
    n_grid: int,
) -> tuple:
    """Return (pb_freqs, sb_freqs, tb_freqs) in rad/sample."""
    pb_freqs = np.concatenate(
        [
            np.linspace(-pb_norm, 0.0, n_grid) * 2 * np.pi,
            np.linspace(0.0, pb_norm, n_grid) * 2 * np.pi,
        ]
    )
    sb_freqs = np.concatenate(
        [
            np.linspace(-0.5, -sb_norm, n_grid) * 2 * np.pi,
            np.linspace(sb_norm, 0.5, n_grid) * 2 * np.pi,
        ]
    )
    tb_freqs_pos = np.linspace(tb_start_norm, sb_norm, n_grid // 2) * 2 * np.pi
    tb_freqs_neg = np.linspace(-sb_norm, -tb_start_norm, n_grid // 2) * 2 * np.pi
    tb_freqs = np.concatenate([tb_freqs_neg, tb_freqs_pos])
    return pb_freqs, sb_freqs, tb_freqs


def _build_tb_desired(
    Hd_pb: np.ndarray,
    delta_s: float,
    n_grid: int,
) -> tuple:
    """Return (tb_desired, mag_start) for a cosine-tapered transition band."""
    mag_start = float(np.max(np.abs(Hd_pb)))
    t = np.linspace(0.0, 1.0, n_grid // 2)
    tb_pos = mag_start + (delta_s - mag_start) * 0.5 * (1 - np.cos(np.pi * t))
    tb_desired = np.concatenate([tb_pos[::-1], tb_pos])
    return tb_desired, mag_start


def _save_and_plot(
    h: np.ndarray,
    pb_freqs: np.ndarray,
    tb_freqs: np.ndarray,
    Hd_pb: np.ndarray,
    tb_desired: np.ndarray,
    f_pass_mhz: float,
    f_stop_mhz: float,
    fs_mhz: float,
    pb_norm: float,
    N: int,
    delta_p: float,
    delta_s: float,
    out_dir: str,
) -> None:
    """Save coefficients, plot filter response, and write in-band error figures."""
    _, H_eqz, H_combined, hc_freq_axis = compute_aaf_responses(
        f_pass_mhz=f_pass_mhz, fs_mhz=fs_mhz
    )
    dc_idx = len(H_eqz) // 2
    H_eqz = H_eqz / H_eqz[dc_idx]
    H_combined = H_combined / H_combined[dc_idx]

    _all_Hd_freqs = np.concatenate([pb_freqs, tb_freqs])
    _all_Hd = np.concatenate([Hd_pb, tb_desired.astype(complex)])
    _sort = np.argsort(_all_Hd_freqs)
    Hd_freqs_plot = _all_Hd_freqs[_sort]
    Hd_plot = _all_Hd[_sort]

    save_coefficients(h, out_dir=out_dir, name="eqz_lowpass_remez")
    plot_response(
        h,
        fs=fs_mhz,
        out_dir=out_dir,
        title=f"EQZ Lowpass FIR (Remez)  N={N}  ±{f_pass_mhz} MHz",
        passband_edges=(0.0, pb_norm),
        freq_unit="MHz",
        Hd_freqs_rad=Hd_freqs_plot,
        Hd=Hd_plot,
        Hc_freqs_mhz=hc_freq_axis,
        Hc=H_combined,
        h_label="complex remez",
    )

    pb_mask = np.abs(hc_freq_axis) <= f_pass_mhz
    freq_pb = hc_freq_axis[pb_mask]
    ref_db = 20 * np.log10(np.abs(H_eqz[pb_mask]) + 1e-12)

    hc_freqs_rad = hc_freq_axis / fs_mhz * 2 * np.pi
    H_rmz_hc = evaluate_response(h, hc_freqs_rad)
    H_rmz_hc = H_rmz_hc / H_rmz_hc[dc_idx]

    def _err_db(H):
        return np.abs(20 * np.log10(np.abs(H[pb_mask]) + 1e-12) - ref_db)

    rmz_err = _err_db(H_rmz_hc)
    print(
        f"  In-band dB magnitude error  peak : {rmz_err.max():.3f} dB"
        f"  (at {freq_pb[np.argmax(rmz_err)]:.1f} MHz)"
    )
    print(f"  In-band dB magnitude error  RMS  : {rmz_err.mean():.3f} dB")

    FSIZE = 20
    curves = [
        (_err_db(H_combined), "direct combine"),
        (_err_db(H_rmz_hc), "complex remez"),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    for err, label in curves:
        ax.plot(freq_pb, err, lw=2.0, label=label)
    ax.set_xlabel("Frequency (MHz)", fontsize=FSIZE)
    ax.set_ylabel("Magnitude error (dB)", fontsize=FSIZE)
    ax.tick_params(labelsize=FSIZE)
    ax.legend(fontsize=FSIZE)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    err_png = os.path.join(out_dir, "inband_error.png")
    fig.savefig(err_png, dpi=150)
    plt.close(fig)
    print(f"Saved in-band error plot → {err_png}")

    pfig = go.Figure()
    for err, label in curves:
        pfig.add_trace(
            go.Scatter(
                x=freq_pb.tolist(), y=err.tolist(), name=label, line=dict(width=2)
            )
        )
    pfig.update_layout(
        xaxis_title="Frequency (MHz)",
        yaxis_title="Magnitude error (dB)",
        hovermode="x unified",
        font=dict(size=FSIZE),
    )
    err_html = os.path.join(out_dir, "inband_error.html")
    pfig.write_html(err_html)
    print(f"Saved in-band error HTML  → {err_html}")

    _write_params_tex(
        path=os.path.join(_HERE, "document", "params_remez.tex"),
        N=N,
        fs_mhz=fs_mhz,
        f_pass_mhz=f_pass_mhz,
        f_stop_mhz=f_stop_mhz,
        delta_p=delta_p,
        delta_s=delta_s,
    )


# ---------------------------------------------------------------------------
# Main design routine
# ---------------------------------------------------------------------------


def design_eqz_lowpass_remez(
    N: int = 63,
    fs_mhz: float = 750.0,
    f_pass_mhz: float = 162.5,
    f_stop_mhz: float = 250.0,
    tb_gap_mhz: float = 0.0,
    delta_p: float = 0.05,
    delta_s: float = 0.10,
    n_iter: int = 150,
    n_grid: int = 512,
    out_dir: str = "gen_rx_filter/figure/eqz_lowpass_remez",
    verbose: bool = False,
    eqz_resp_path: Optional[str] = None,
    use_transition_band: bool = True,
) -> np.ndarray:
    """Design a complex lowpass FIR matched to the UCDC equalizer response.

    Uses the complex Remez algorithm (Lawson IRLS) instead of a CVXPY SOCP.
    The band weight ratio delta_p/delta_s is passed to the solver so that the
    equiripple condition at convergence targets both tolerances simultaneously.

    Args:
        N: Number of filter taps.
        fs_mhz: Sample rate in MHz.
        f_pass_mhz: Passband half-bandwidth in MHz (±f_pass_mhz).
        f_stop_mhz: Stopband start frequency in MHz.  Moving this outward widens
            the transition band and is the primary lever for reducing in-band error.
        tb_gap_mhz: Gap between the passband edge and the first transition-band
            constraint point (MHz).  Leaving a small unconstrained gap just above
            the passband edge reduces the "pull" on the passband near its edge,
            further lowering in-band error.  Values above ~20 MHz start to produce
            a gain bump in the gap region.
        delta_p: Passband ripple tolerance (linear, relative to DC gain).
        delta_s: Stopband attenuation tolerance (linear).
        n_iter: Maximum Lawson iterations.
        n_grid: Frequency grid points per band.
        out_dir: Output directory for saved files.
        verbose: Print per-iteration solver diagnostics.

    Returns:
        Designed complex tap vector h of shape (N,).
    """
    if eqz_resp_path is not None:
        eqz_resp = load_freq_resp_as_fft(eqz_resp_path, n_fft=256, fs_mhz=fs_mhz)
    else:
        eqz_resp = gen_eqz_freq_resp(sig=DistortedSig.UCDC, n_taps=4)
    dc_gain = eqz_resp[0]
    eqz_resp = eqz_resp / dc_gain

    pb_norm = f_pass_mhz / fs_mhz
    sb_norm = f_stop_mhz / fs_mhz
    tb_start_norm = pb_norm + tb_gap_mhz / fs_mhz
    pb_freqs, sb_freqs, tb_freqs = _build_frequency_grids(pb_norm, sb_norm, tb_start_norm, n_grid)

    Hd_pb = interp_fft_response(eqz_resp, fs=fs_mhz, target_freqs_rad=pb_freqs)
    tb_desired, mag_start = _build_tb_desired(Hd_pb, delta_s, n_grid)

    # sb_weight = delta_p/delta_s encodes the relative tolerance so that at
    # convergence: peak_pb_error ≈ delta_p and peak_sb_level ≈ delta_s.
    sb_weight = delta_p / delta_s

    print(f"Designing {N}-tap complex lowpass FIR  [complex Remez / Lawson IRLS]")
    print(f"  fs          = {fs_mhz} MHz")
    print(f"  Passband    = ±{f_pass_mhz} MHz  (±{pb_norm:.4f} × fs)")
    print(f"  Stopband    ≥  {f_stop_mhz} MHz  ( {sb_norm:.4f} × fs)")
    if eqz_resp_path is not None:
        print(f"  Desired     = {eqz_resp_path}, 256-bin FFT, DC-normalised")
    else:
        print("  Desired     = eqz_resp (UCDC, n_taps=4), 256-bin FFT, DC-normalised")
    print(
        f"  DC gain removed: |{np.abs(dc_gain):.6f}|  ∠{np.degrees(np.angle(dc_gain)):.2f}°"
    )
    print(f"  delta_p     = {delta_p}  (passband),  delta_s = {delta_s}  (stopband)")
    print(f"  sb_weight   = delta_p/delta_s = {sb_weight:.4f}")
    if use_transition_band:
        print(
            f"  transition  = cosine taper  [{20*np.log10(mag_start+1e-12):.2f} dB → {20*np.log10(delta_s+1e-12):.2f} dB]"
            f"  gap={tb_gap_mhz} MHz  ({len(tb_freqs)} pts)"
        )
    else:
        print("  transition  = disabled")
    print(f"  max_iter    = {n_iter}")

    h = design_complex_fir_remez(
        N=N,
        passband_freqs=pb_freqs,
        stopband_freqs=sb_freqs,
        Hd_passband=Hd_pb,
        sb_weight=sb_weight,
        transition_freqs=tb_freqs if use_transition_band else None,
        Hd_transition=tb_desired if use_transition_band else None,
        tb_weight=0.05,
        n_iter=n_iter,
        verbose=verbose,
    )

    H_sb = evaluate_response(h, sb_freqs)
    sb_pk = float(np.max(np.abs(H_sb)))
    print(f"\n  Stopband peak |H| : {sb_pk:.5f}  ({20 * np.log10(sb_pk + 1e-12):.1f} dB)")

    _save_and_plot(
        h=h,
        pb_freqs=pb_freqs,
        tb_freqs=tb_freqs,
        Hd_pb=Hd_pb,
        tb_desired=tb_desired,
        f_pass_mhz=f_pass_mhz,
        f_stop_mhz=f_stop_mhz,
        fs_mhz=fs_mhz,
        pb_norm=pb_norm,
        N=N,
        delta_p=delta_p,
        delta_s=delta_s,
        out_dir=out_dir,
    )
    return h


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Design a complex lowpass FIR via the complex Remez algorithm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--N", type=int, default=63, help="Number of filter taps")
    p.add_argument("--fs", type=float, default=750.0, help="Sample rate in MHz")
    p.add_argument(
        "--f-pass",
        type=float,
        default=162.5,  # 162.5
        help="Passband half-bandwidth in MHz",
    )
    p.add_argument(
        "--f-stop",
        type=float,
        default=210.0,
        help="Stopband start frequency in MHz",
    )
    p.add_argument(
        "--tb-gap",
        type=float,
        default=20.0,
        help="Gap between passband edge and TB constraint start (MHz)",
    )
    p.add_argument(
        "--delta-p",
        type=float,
        default=0.05,
        help="Passband ripple tolerance (sets sb_weight = delta_p/delta_s)",
    )
    p.add_argument(
        "--delta-s",
        type=float,
        default=0.1,
        help="Stopband attenuation tolerance",
    )
    p.add_argument("--n-iter", type=int, default=150, help="Maximum Lawson iterations")
    p.add_argument(
        "--n-grid", type=int, default=512, help="Frequency grid points per band"
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(_HERE, "figure", "eqz_lowpass_remez"),
        help="Output directory",
    )
    p.add_argument(
        "--verbose", action="store_true", help="Print per-iteration Lawson diagnostics"
    )
    p.add_argument(
        "--no-transition-band",
        action="store_true",
        # action="store_false",
        help="Disable transition band constraints; solver uses only passband and stopband",
    )
    p.add_argument(
        "--eqz-resp-path",
        type=str,
        # default=None,
        default="trx_board_meas/freq_resp/rx_ch1_h3.npy",
        help="Path to a .npy freq response file (shape (2,N) complex128) to use as eqz_resp "
             "instead of the default UCDC model. E.g. trx_board_meas/freq_resp/rx_ch1_h3.npy",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    design_eqz_lowpass_remez(
        N=args.N,
        fs_mhz=args.fs,
        f_pass_mhz=args.f_pass,
        f_stop_mhz=args.f_stop,
        tb_gap_mhz=args.tb_gap,
        delta_p=args.delta_p,
        delta_s=args.delta_s,
        n_iter=args.n_iter,
        n_grid=args.n_grid,
        out_dir=args.out_dir,
        verbose=args.verbose,
        eqz_resp_path=args.eqz_resp_path,
        use_transition_band=not args.no_transition_band,
    )

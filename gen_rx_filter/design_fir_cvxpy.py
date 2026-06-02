"""
Complex-valued FIR filter design via convex optimisation (CVXPY).

The filter h[n] is fully complex with no symmetry constraint. The DTFT is:

    H(w) = sum_{n=0}^{N-1} h[n] * exp(-j * w * n)

Sampled over a frequency grid this becomes the linear map:

    H = A @ h,   A[k, n] = exp(-j * w[k] * n)

Two optimisation modes are provided:

  minimax  – minimise the worst-case deviation δ jointly over passband
             and stopband:
               minimise   δ
               subject to |A_pb @ h - Hd_pb| ≤ δ   (passband error)
                          |A_sb @ h|          ≤ δ   (stopband leakage)

  fixed    – feasibility / minimum-error problem with separate tolerances:
               minimise   Σ |A_pb @ h - Hd_pb|
               subject to |A_pb @ h - Hd_pb| ≤ delta_p
                          |A_sb @ h|          ≤ delta_s

Both problems are convex (second-order cone programs) and are handled
natively by CVXPY using its complex-variable support.
"""

import argparse
import os
from typing import Optional

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Core matrix builder
# ---------------------------------------------------------------------------

def build_frequency_matrix(freqs: np.ndarray, N: int) -> np.ndarray:
    """Build the evaluation matrix A[k, n] = exp(-j * freqs[k] * n).

    This maps tap coefficients h ∈ ℂ^N to frequency samples H = A @ h ∈ ℂ^K,
    which is the DTFT of h evaluated at the requested angular frequencies.

    Args:
        freqs: Angular frequencies in radians/sample, shape (K,).
        N: Number of filter taps.

    Returns:
        Complex ndarray of shape (K, N).
    """
    # outer product gives phase increments (K, N): freqs[k] * n[n]
    return np.exp(-1j * np.outer(freqs, np.arange(N)))


# ---------------------------------------------------------------------------
# CVXPY optimisation
# ---------------------------------------------------------------------------

def design_complex_fir_cvxpy(
    N: int,
    passband_freqs: np.ndarray,
    stopband_freqs: np.ndarray,
    Hd_passband: np.ndarray,
    delta_p: Optional[float] = None,
    delta_s: Optional[float] = None,
    mode: str = 'minimax',
    lambda_reg: float = 0.0,
    transition_freqs: Optional[np.ndarray] = None,
    delta_t: float = 1.0,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Design a complex-valued FIR filter with CVXPY.

    Args:
        N: Number of filter taps.
        passband_freqs: Angular frequencies (rad/sample) in the passband, shape (Kp,).
        stopband_freqs: Angular frequencies (rad/sample) in the stopband, shape (Ks,).
        Hd_passband: Desired complex response at passband_freqs, shape (Kp,).
        delta_p: Passband ripple tolerance; used only in 'fixed' mode.
        delta_s: Stopband attenuation tolerance; used only in 'fixed' mode.
        mode: 'minimax' or 'fixed' (see module docstring).
        lambda_reg: L2 regularisation weight on tap energy ‖h‖².
            Adding a small positive value (e.g. 1e-3) penalises large taps.
            Set to 0 (default) to disable.
        transition_freqs: Angular frequencies (rad/sample) in the transition
            band, shape (Kt,). When provided, an upper-magnitude constraint
            |A_tb @ h| ≤ delta_t is added, preventing overshoot in the gap
            between passband and stopband.
        delta_t: Upper magnitude bound for the transition band.  Scalar for a
            flat cap, or ndarray of the same length as transition_freqs for a
            per-frequency bound (e.g. a taper from the passband edge value
            down to near delta_s at the stopband edge).  Ignored when
            transition_freqs is None.
        solver: CVXPY solver name (e.g. 'CLARABEL', 'SCS'). None → auto-select.
        verbose: Forward verbose flag to the CVXPY solver.

    Returns:
        Complex tap vector h of shape (N,).

    Raises:
        ValueError: If the solver fails or the problem is infeasible.
    """
    A_pb = build_frequency_matrix(passband_freqs, N)   # (Kp, N) – passband rows
    A_sb = build_frequency_matrix(stopband_freqs, N)   # (Ks, N) – stopband rows

    # CVXPY complex variable h ∈ ℂ^N; no symmetry constraint is imposed.
    h = cp.Variable(N, complex=True)

    # L2 regularisation on tap energy: penalises ||h||² to prevent the solver
    # from using arbitrarily large coefficients that cause transition-band spikes.
    reg = lambda_reg * (cp.sum_squares(cp.real(h)) + cp.sum_squares(cp.imag(h)))

    if mode == 'minimax':
        # Single slack δ bounds both the passband tracking error and the
        # stopband leakage. cp.abs() on a complex expression yields the modulus
        # and is recognised by CVXPY as a convex SOC constraint.
        delta = cp.Variable(nonneg=True)
        constraints = [
            cp.abs(A_pb @ h - Hd_passband) <= delta,   # passband: match Hd
            cp.abs(A_sb @ h) <= delta,                  # stopband: suppress
        ]
        objective = cp.Minimize(delta + reg)

    elif mode == 'fixed':
        if delta_p is None or delta_s is None:
            raise ValueError("'fixed' mode requires both delta_p and delta_s.")
        # Hard constraints on per-band tolerances; minimise total passband
        # error so the objective is bounded and the solver has clear direction.
        pb_error = A_pb @ h - Hd_passband
        constraints = [
            cp.abs(pb_error) <= delta_p,    # |H(w) - Hd(w)| ≤ δ_p  ∀w ∈ PB
            cp.abs(A_sb @ h) <= delta_s,    # |H(w)|         ≤ δ_s  ∀w ∈ SB
        ]
        objective = cp.Minimize(cp.sum(cp.abs(pb_error)) + reg)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'minimax' or 'fixed'.")

    # Optional transition-band upper-magnitude constraint.
    # Prevents the solver exploiting the unconstrained gap between passband and
    # stopband to put large energy there while satisfying the band constraints.
    if transition_freqs is not None:
        A_tb = build_frequency_matrix(transition_freqs, N)
        constraints.append(cp.abs(A_tb @ h) <= delta_t)

    prob = cp.Problem(objective, constraints)
    solve_kwargs = {'verbose': verbose}
    if solver is not None:
        solve_kwargs['solver'] = solver
    prob.solve(**solve_kwargs)

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        raise ValueError(
            f"CVXPY solver did not find an optimal solution (status: {prob.status}).\n"
            "Try a different solver (--solver SCS) or relax the tolerances."
        )
    if prob.status == 'optimal_inaccurate':
        print("Warning: solver returned 'optimal_inaccurate'; result may be approximate.")

    return h.value   # complex ndarray (N,)


# ---------------------------------------------------------------------------
# Frequency response evaluation
# ---------------------------------------------------------------------------

def evaluate_response(h: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Evaluate H(w) = Σ h[n] exp(-j w n) at arbitrary angular frequencies.

    Args:
        h: Complex tap coefficients, shape (N,).
        freqs: Angular frequencies (rad/sample), shape (K,).

    Returns:
        Complex frequency response, shape (K,).
    """
    return build_frequency_matrix(freqs, len(h)) @ h


def interp_fft_response(
    fft_resp: np.ndarray,
    fs: float,
    target_freqs_rad: np.ndarray,
) -> np.ndarray:
    """Interpolate a two-sided FFT frequency response at arbitrary angular frequencies.

    The input follows standard FFT bin ordering (DC first, then positive bins,
    then negative bins). Real and imaginary parts are interpolated independently
    using linear interpolation.

    Args:
        fft_resp: Complex FFT output of length n_fft, in FFT bin order.
        fs: Sample rate in the same units used to produce fft_resp (e.g. MHz).
            Only used to build the bin-frequency axis; the returned values are
            independent of this unit.
        target_freqs_rad: Target angular frequencies in radians/sample, shape (K,).
            Values must lie within (-π, π].

    Returns:
        Complex interpolated response, shape (K,).
    """
    n_fft = len(fft_resp)
    bin_freqs_rad = 2 * np.pi * np.fft.fftfreq(n_fft)

    order = np.argsort(bin_freqs_rad)
    sorted_freqs = bin_freqs_rad[order]
    sorted_mag   = np.abs(fft_resp[order])
    sorted_phase = np.unwrap(np.angle(fft_resp[order]))

    # Interpolate magnitude and unwrapped phase separately.
    # Interpolating real/imag directly is inaccurate when phase rotates fast
    # between bins (e.g. a linear-phase trend), producing wrong magnitudes.
    interp_mag   = interp1d(sorted_freqs, sorted_mag,   kind='linear',
                            bounds_error=False,
                            fill_value=(sorted_mag[0],   sorted_mag[-1]))
    interp_phase = interp1d(sorted_freqs, sorted_phase, kind='linear',
                            bounds_error=False,
                            fill_value=(sorted_phase[0], sorted_phase[-1]))
    return interp_mag(target_freqs_rad) * np.exp(1j * interp_phase(target_freqs_rad))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_response(
    h: np.ndarray,
    fs: float = 1.0,
    out_dir: str = '.',
    title: str = 'Complex FIR Filter',
    n_fft: int = 2048,
    passband_edges: Optional[tuple] = None,
    freq_unit: str = 'Hz',
    Hd_freqs_rad: Optional[np.ndarray] = None,
    Hd: Optional[np.ndarray] = None,
    h_label: str = 'cvx. opt.',
) -> None:
    """Plot magnitude (dB) and phase responses and save to PNG and interactive HTML.

    The designed response H(w) is shown as a continuous line over the full
    spectrum. When Hd_freqs_rad and Hd are provided, the desired response is
    overlaid as a dashed line on the same axes so the match can be inspected.

    Uses scipy.signal.freqz for a dense, accurate frequency grid.

    Args:
        h: Complex tap coefficients, shape (N,).
        fs: Sample rate in the unit given by freq_unit (for axis labels).
        out_dir: Directory where outputs are saved.
        title: Figure title.
        n_fft: FFT size for the frequency grid.
        passband_edges: Optional (low, high) normalised freq pair to shade.
        freq_unit: Unit string for the frequency axis label (e.g. 'Hz', 'MHz').
        Hd_freqs_rad: Angular frequencies (rad/sample) at which Hd is defined,
            shape (K,). Convert to axis units via f = w * fs / (2π).
        Hd: Desired complex response at Hd_freqs_rad, shape (K,). Both
            Hd_freqs_rad and Hd must be supplied together.
    """
    os.makedirs(out_dir, exist_ok=True)

    # scipy.signal.freqz handles complex b coefficients correctly
    w, H = signal.freqz(h, worN=n_fft, whole=True, fs=fs)
    # Shift to [-fs/2, +fs/2] for a centred display
    w = np.fft.fftshift(w - fs * (w >= fs / 2))
    H = np.fft.fftshift(H)

    mag_db    = 20 * np.log10(np.abs(H) + 1e-12)
    phase_deg = np.degrees(np.unwrap(np.angle(H)))

    # Convert desired-response frequencies from rad/sample → axis units
    hd_freq_axis = hd_mag_db = hd_phase_deg = None
    if Hd_freqs_rad is not None and Hd is not None:
        hd_freq_axis  = Hd_freqs_rad * fs / (2 * np.pi)
        hd_mag_db     = 20 * np.log10(np.abs(Hd) + 1e-12)
        hd_phase_deg  = np.degrees(np.unwrap(np.angle(Hd)))

    # --- matplotlib PNG (magnitude only) ---
    FSIZE = 20
    fig, ax = plt.subplots(figsize=(12, 5))

    if hd_freq_axis is not None:
        ax.plot(hd_freq_axis, hd_mag_db, lw=2.0, label='desired resp.', zorder=2)
    ax.plot(w, mag_db, lw=1.5, label=h_label, zorder=4)
    ax.set_ylabel('Magnitude (dB)', fontsize=FSIZE)
    ax.set_xlabel(f'Frequency ({freq_unit})', fontsize=FSIZE)
    ax.tick_params(labelsize=FSIZE)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=FSIZE)
    if passband_edges is not None:
        pb_lo, pb_hi = passband_edges
        for lo, hi in [(-pb_hi * fs, -pb_lo * fs), (pb_lo * fs, pb_hi * fs)]:
            ax.axvspan(lo, hi, alpha=0.10, color='green')

    fig.tight_layout()
    png_path = os.path.join(out_dir, 'filter_response.png')
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {png_path}")

    # --- zoomed-in magnitude PNG (in-band) ---
    if passband_edges is not None:
        pb_lo, pb_hi = passband_edges
        x_lo = -pb_hi * fs - 5
        x_hi =  pb_hi * fs + 5
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        if hd_freq_axis is not None:
            ax2.plot(hd_freq_axis, hd_mag_db, lw=2.0, label='desired resp.', zorder=2)
        ax2.plot(w, mag_db, lw=1.5, label=h_label, zorder=4)
        for lo, hi in [(-pb_hi * fs, -pb_lo * fs), (pb_lo * fs, pb_hi * fs)]:
            ax2.axvspan(lo, hi, alpha=0.10, color='green')
        ax2.set_xlim(x_lo, x_hi)
        ax2.set_ylim(-2.0, 2.0)
        ax2.set_ylabel('Magnitude (dB)', fontsize=FSIZE)
        ax2.set_xlabel(f'Frequency ({freq_unit})', fontsize=FSIZE)
        ax2.tick_params(labelsize=FSIZE)
        ax2.grid(True, alpha=0.4)
        ax2.legend(fontsize=FSIZE)
        fig2.tight_layout()
        zoom_path = os.path.join(out_dir, 'filter_response_inband.png')
        fig2.savefig(zoom_path, dpi=150)
        plt.close(fig2)
        print(f"Saved zoomed plot → {zoom_path}")

    # --- Plotly interactive HTML ---
    pfig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.12,
    )
    if hd_freq_axis is not None:
        pfig.add_trace(go.Scatter(x=hd_freq_axis.tolist(), y=hd_mag_db.tolist(),
                                  name='desired resp.', line=dict(width=2.0)),
                       row=1, col=1)
        pfig.add_trace(go.Scatter(x=hd_freq_axis.tolist(), y=hd_phase_deg.tolist(),
                                  name='desired resp.', line=dict(width=2.0),
                                  showlegend=False),
                       row=2, col=1)
    pfig.add_trace(go.Scatter(x=w.tolist(), y=mag_db.tolist(),
                              name=h_label, line=dict(width=1.5)),
                   row=1, col=1)
    pfig.add_trace(go.Scatter(x=w.tolist(), y=phase_deg.tolist(),
                              name=h_label, line=dict(width=1.5),
                              showlegend=False),
                   row=2, col=1)

    if passband_edges is not None:
        pb_lo, pb_hi = passband_edges
        for row in (1, 2):
            for lo, hi in [(-pb_hi * fs, -pb_lo * fs), (pb_lo * fs, pb_hi * fs)]:
                pfig.add_vrect(x0=lo, x1=hi, fillcolor='green', opacity=0.10,
                               line_width=0, row=row, col=1)

    freq_label = f'Frequency ({freq_unit})'
    pfig.update_xaxes(title_text=freq_label, title_font=dict(size=FSIZE),
                      tickfont=dict(size=FSIZE))
    pfig.update_yaxes(title_font=dict(size=FSIZE), tickfont=dict(size=FSIZE))
    pfig.update_yaxes(title_text='Magnitude (dB)', row=1, col=1)
    pfig.update_yaxes(title_text='Phase (degree)', row=2, col=1)
    pfig.update_layout(hovermode='x unified', font=dict(size=FSIZE))
    html_path = os.path.join(out_dir, 'filter_response.html')
    pfig.write_html(html_path)
    print(f"Saved interactive plot → {html_path}")


# ---------------------------------------------------------------------------
# Coefficient I/O
# ---------------------------------------------------------------------------

def save_coefficients(
    h: np.ndarray,
    out_dir: str = '.',
    name: str = 'fir_coef',
) -> None:
    """Save tap coefficients to .npy and .csv files.

    The CSV has two columns: real and imaginary parts.

    Args:
        h: Complex tap coefficients, shape (N,).
        out_dir: Output directory.
        name: Base filename without extension.
    """
    os.makedirs(out_dir, exist_ok=True)

    npy_path = os.path.join(out_dir, f'{name}.npy')
    np.save(npy_path, h)
    print(f"Saved .npy → {npy_path}")

    csv_path = os.path.join(out_dir, f'{name}.csv')
    np.savetxt(
        csv_path,
        np.column_stack([h.real, h.imag]),
        delimiter=',',
        header='real,imag',
        comments='',
        fmt='%.10f',
    )
    print(f"Saved .csv → {csv_path}")


# ---------------------------------------------------------------------------
# Minimal bandpass example
# ---------------------------------------------------------------------------

def bandpass_example(
    N: int = 31,
    fs: float = 1.0,
    pb_low: float = 0.10,
    pb_high: float = 0.20,
    sb_low: float = 0.00,
    sb_high: float = 0.07,
    sb_low2: float = 0.26,
    sb_high2: float = 0.50,
    delta_p: float = 0.05,
    delta_s: float = 0.01,
    mode: str = 'minimax',
    n_grid: int = 256,
    solver: Optional[str] = None,
    out_dir: str = 'cvxpy_fir_output',
    verbose: bool = False,
) -> np.ndarray:
    """Design and save a complex bandpass FIR filter.

    Passband: [pb_low, pb_high] (normalised frequency, 0–0.5).
    Stopband: [sb_low, sb_high] ∪ [sb_low2, sb_high2].

    The desired passband response is a linear-phase target:

        Hd(w) = exp(-j * w * (N-1)/2)

    which centres the group delay at the midpoint of the tap vector,
    giving a causal filter with a delay of (N-1)/2 samples.

    Args:
        N: Number of taps.
        fs: Sample rate in Hz (for labelling only).
        pb_low, pb_high: Passband edges (normalised, 0–0.5).
        sb_low, sb_high: First stopband edges (normalised).
        sb_low2, sb_high2: Second stopband edges (normalised).
        delta_p: Passband tolerance (fixed mode).
        delta_s: Stopband tolerance (fixed mode).
        mode: 'minimax' or 'fixed'.
        n_grid: Frequency grid points per band.
        solver: CVXPY solver name or None for auto.
        out_dir: Output directory.
        verbose: Verbose solver output.

    Returns:
        Designed complex tap vector h of shape (N,).
    """
    # Build frequency grids (angular: multiply normalised freq by 2π)
    pb_freqs = np.linspace(pb_low, pb_high, n_grid) * 2 * np.pi
    sb_freqs = np.concatenate([
        np.linspace(sb_low, sb_high, n_grid) * 2 * np.pi,
        np.linspace(sb_low2, sb_high2, n_grid) * 2 * np.pi,
    ])

    # Linear-phase desired response: pure delay of (N-1)/2 samples
    group_delay = (N - 1) / 2.0
    Hd_pb = np.exp(-1j * pb_freqs * group_delay)

    print(f"Designing {N}-tap complex bandpass FIR  [mode={mode}]")
    print(f"  Passband : [{pb_low:.3f}, {pb_high:.3f}] × fs")
    print(f"  Stopband : [{sb_low:.3f}, {sb_high:.3f}] ∪ [{sb_low2:.3f}, {sb_high2:.3f}] × fs")

    h = design_complex_fir_cvxpy(
        N=N,
        passband_freqs=pb_freqs,
        stopband_freqs=sb_freqs,
        Hd_passband=Hd_pb,
        delta_p=delta_p,
        delta_s=delta_s,
        mode=mode,
        solver=solver,
        verbose=verbose,
    )

    # Report achieved errors on the optimisation grid
    H_pb = evaluate_response(h, pb_freqs)
    H_sb = evaluate_response(h, sb_freqs)
    pb_err = np.max(np.abs(H_pb - Hd_pb))
    sb_err = np.max(np.abs(H_sb))
    print(f"  Achieved passband peak error : {pb_err:.4f}")
    print(f"  Achieved stopband peak level : {sb_err:.4f}  ({20*np.log10(sb_err+1e-12):.1f} dB)")

    save_coefficients(h, out_dir=out_dir, name='complex_bandpass')
    plot_response(
        h, fs=fs, out_dir=out_dir,
        title=f'Complex Bandpass FIR  N={N}  mode={mode}',
        passband_edges=(pb_low, pb_high),
    )
    return h


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Design a complex FIR filter via CVXPY convex optimisation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--N', type=int, default=31,
                   help='Number of filter taps')
    p.add_argument('--fs', type=float, default=1.0,
                   help='Sample rate in Hz (axis labels only)')
    p.add_argument('--pb-low', type=float, default=0.10,
                   help='Passband lower edge (normalised 0–0.5)')
    p.add_argument('--pb-high', type=float, default=0.20,
                   help='Passband upper edge (normalised 0–0.5)')
    p.add_argument('--sb-low', type=float, default=0.00,
                   help='Stopband lower edge (normalised 0–0.5)')
    p.add_argument('--sb-high', type=float, default=0.07,
                   help='Stopband upper edge (normalised 0–0.5)')
    p.add_argument('--sb-low2', type=float, default=0.26,
                   help='Second stopband lower edge (normalised 0–0.5)')
    p.add_argument('--sb-high2', type=float, default=0.50,
                   help='Second stopband upper edge (normalised 0–0.5)')
    p.add_argument('--delta-p', type=float, default=0.05,
                   help='Passband ripple tolerance (fixed mode)')
    p.add_argument('--delta-s', type=float, default=0.01,
                   help='Stopband attenuation tolerance (fixed mode)')
    p.add_argument('--mode', choices=['minimax', 'fixed'], default='minimax',
                   help='Optimisation mode')
    p.add_argument('--n-grid', type=int, default=256,
                   help='Frequency grid points per band')
    p.add_argument('--solver', type=str, default=None,
                   help='CVXPY solver name (CLARABEL, SCS, …). Default: auto')
    p.add_argument('--out-dir', type=str, default='cvxpy_fir_output',
                   help='Output directory')
    p.add_argument('--verbose', action='store_true',
                   help='Show solver output')
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()
    bandpass_example(
        N=args.N,
        fs=args.fs,
        pb_low=args.pb_low,
        pb_high=args.pb_high,
        sb_low=args.sb_low,
        sb_high=args.sb_high,
        sb_low2=args.sb_low2,
        sb_high2=args.sb_high2,
        delta_p=args.delta_p,
        delta_s=args.delta_s,
        mode=args.mode,
        n_grid=args.n_grid,
        solver=args.solver,
        out_dir=args.out_dir,
        verbose=args.verbose,
    )

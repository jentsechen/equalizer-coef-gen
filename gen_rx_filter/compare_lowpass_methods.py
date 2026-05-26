"""
Overlay comparison of three curves:
  1. Desired response  – eqz_resp (UCDC, n_taps=4), DC-normalised
  2. CVX lowpass       – CVXPY fixed mode with transition-band constraint
                         (matches the default settings of run_eqz_lowpass.py)
  3. Remez lowpass     – complex Lawson IRLS, transition band unconstrained
                         (matches the default settings of run_eqz_lowpass_remez.py)

Run from the repository root:
    python3.8 gen_rx_filter/compare_lowpass_methods.py
    python3.8 gen_rx_filter/compare_lowpass_methods.py --N 95 --n-grid 512
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from gen_rx_filter import gen_eqz_freq_resp, DistortedSig
from design_fir_cvxpy import (
    design_complex_fir_cvxpy,
    evaluate_response,
    interp_fft_response,
    build_frequency_matrix,
)
from run_eqz_lowpass_remez import design_complex_fir_remez


# ---------------------------------------------------------------------------
# Shared problem setup
# ---------------------------------------------------------------------------

def _build_problem(
    fs_mhz: float = 750.0,
    f_pass_mhz: float = 162.5,
    f_stop_mhz: float = 250.0,
    n_grid: int = 512,
):
    """Build the shared frequency grids and desired response."""
    eqz_resp = gen_eqz_freq_resp(sig=DistortedSig.UCDC, n_taps=4)
    eqz_resp = eqz_resp / eqz_resp[0]   # DC-normalise

    pb_norm = f_pass_mhz / fs_mhz
    sb_norm = f_stop_mhz / fs_mhz

    pb_freqs = np.concatenate([
        np.linspace(-pb_norm, 0.0,     n_grid) * 2 * np.pi,
        np.linspace( 0.0,     pb_norm, n_grid) * 2 * np.pi,
    ])
    sb_freqs = np.concatenate([
        np.linspace(-0.5,    -sb_norm, n_grid) * 2 * np.pi,
        np.linspace( sb_norm, 0.5,     n_grid) * 2 * np.pi,
    ])

    Hd_pb = interp_fft_response(eqz_resp, fs=fs_mhz, target_freqs_rad=pb_freqs)

    # Dense desired-response axis from the raw 256-bin eqz_resp
    n_fft_des     = len(eqz_resp)
    des_freqs_mhz = np.fft.fftshift(np.fft.fftfreq(n_fft_des, d=1.0 / fs_mhz))
    des_resp      = np.fft.fftshift(eqz_resp)

    return dict(
        pb_freqs=pb_freqs, sb_freqs=sb_freqs,
        Hd_pb=Hd_pb,
        pb_norm=pb_norm, sb_norm=sb_norm,
        des_freqs_mhz=des_freqs_mhz, des_resp=des_resp,
        eqz_resp=eqz_resp,
    )


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_comparison(
    N: int = 63,
    fs_mhz: float = 750.0,
    f_pass_mhz: float = 162.5,
    f_stop_mhz: float = 250.0,
    delta_p: float = 0.05,
    delta_s: float = 0.10,
    lambda_reg: float = 1e-3,
    n_grid: int = 512,
    n_iter_remez: int = 150,
    solver_cvx: str = None,
    out_dir: str = None,
    verbose: bool = False,
):
    if out_dir is None:
        out_dir = os.path.join(_HERE, 'figure', 'compare_lowpass_methods')
    os.makedirs(out_dir, exist_ok=True)

    prob = _build_problem(fs_mhz, f_pass_mhz, f_stop_mhz, n_grid)
    pb_freqs      = prob['pb_freqs']
    sb_freqs      = prob['sb_freqs']
    Hd_pb         = prob['Hd_pb']
    pb_norm       = prob['pb_norm']
    sb_norm       = prob['sb_norm']
    des_freqs_mhz = prob['des_freqs_mhz']
    des_resp      = prob['des_resp']

    # ---- CVX fixed mode (no transition-band constraint) ------------------
    print(f"[1/2] CVXPY fixed mode  N={N}  (no transition-band constraint)")
    h_cvx = design_complex_fir_cvxpy(
        N=N,
        passband_freqs=pb_freqs,
        stopband_freqs=sb_freqs,
        Hd_passband=Hd_pb,
        mode='fixed',
        delta_p=delta_p,
        delta_s=delta_s,
        lambda_reg=lambda_reg,
        transition_freqs=None,
        solver=solver_cvx,
        verbose=verbose,
    )

    # ---- Remez (Lawson IRLS) – two-pass stopband calibration -------------
    # Pass 1: initial weight ratio sets the equiripple balance.
    sb_weight = delta_p / delta_s
    print(f"[2/2] Remez Lawson IRLS  N={N}  pass 1  sb_weight={sb_weight:.3f}")
    h_rmz = design_complex_fir_remez(
        N=N,
        passband_freqs=pb_freqs,
        stopband_freqs=sb_freqs,
        Hd_passband=Hd_pb,
        sb_weight=sb_weight,
        n_iter=n_iter_remez,
        verbose=verbose,
    )
    # The equiripple condition gives sb_peak ≈ pb_peak / sb_weight, where
    # pb_peak (δ*) is set by N — not by delta_p.  Recalibrate sb_weight
    # using the measured pb_peak so pass 2 lands sb_peak ≈ delta_s.
    pb_achieved = float(np.max(np.abs(evaluate_response(h_rmz, pb_freqs) - Hd_pb)))
    sb_weight = pb_achieved / delta_s
    print(f"           pass 2  pb_achieved={pb_achieved:.5f}  "
          f"sb_weight_corrected={sb_weight:.3f}")
    h_rmz = design_complex_fir_remez(
        N=N,
        passband_freqs=pb_freqs,
        stopband_freqs=sb_freqs,
        Hd_passband=Hd_pb,
        sb_weight=sb_weight,
        n_iter=n_iter_remez,
        verbose=verbose,
    )

    # ---- Metrics ---------------------------------------------------------
    def _metrics(h, label):
        H_pb  = evaluate_response(h, pb_freqs)
        H_sb  = evaluate_response(h, sb_freqs)
        err   = H_pb - Hd_pb
        pk    = float(np.max(np.abs(err)))
        rms   = float(np.sqrt(np.mean(np.abs(err) ** 2)))
        sb_pk = float(np.max(np.abs(H_sb)))
        print(f"  {label:<16}  pb peak={pk:.5f} ({20*np.log10(pk+1e-12):.1f} dB)"
              f"  pb rms={rms:.5f} ({20*np.log10(rms+1e-12):.1f} dB)"
              f"  sb peak={sb_pk:.5f} ({20*np.log10(sb_pk+1e-12):.1f} dB)")

    print()
    _metrics(h_cvx, 'CVX (fixed+TB)')
    _metrics(h_rmz, 'Remez (IRLS)')

    # ---- Dense frequency axes for plotting -------------------------------
    n_fft = 4096
    w_cvx, H_cvx = signal.freqz(h_cvx, worN=n_fft, whole=True, fs=fs_mhz)
    w_rmz, H_rmz = signal.freqz(h_rmz, worN=n_fft, whole=True, fs=fs_mhz)

    def _shift(w, H):
        w = np.fft.fftshift(w - fs_mhz * (w >= fs_mhz / 2))
        return w, np.fft.fftshift(H)

    w_cvx, H_cvx = _shift(w_cvx, H_cvx)
    w_rmz, H_rmz = _shift(w_rmz, H_rmz)

    mag_des = 20 * np.log10(np.abs(des_resp) + 1e-12)
    mag_cvx = 20 * np.log10(np.abs(H_cvx)   + 1e-12)
    mag_rmz = 20 * np.log10(np.abs(H_rmz)   + 1e-12)

    pb_lo_mhz = -f_pass_mhz
    pb_hi_mhz =  f_pass_mhz
    sb_lo_mhz =  f_stop_mhz

    # ====================================================================
    # Matplotlib: 2-panel figure
    #   row 1 – magnitude (full-band)
    #   row 2 – in-band magnitude zoom (y range −10 to +10 dB)
    # ====================================================================
    FSIZE      = 13
    COLORS     = {'des': '#888888', 'cvx': '#1f77b4', 'rmz': '#ff7f0e'}
    zoom_margin = 15.0

    fig, (ax_mag, ax_zoom) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(
        f'Desired vs CVX (fixed+TB) vs Remez (IRLS)   N={N}   '
        f'±{f_pass_mhz} MHz passband   {f_stop_mhz} MHz stopband',
        fontsize=FSIZE + 1,
    )

    # ---- row 1: full-band magnitude -------------------------------------
    ax_mag.plot(des_freqs_mhz, mag_des, lw=1.5, color=COLORS['des'],
                label='Desired freq. resp.', zorder=1)
    ax_mag.plot(w_cvx, mag_cvx, lw=1.8, color=COLORS['cvx'],
                label='CVX fixed', zorder=3)
    ax_mag.plot(w_rmz, mag_rmz, lw=1.8, color=COLORS['rmz'],
                label='Remez IRLS', zorder=4)
    ax_mag.axvspan(pb_lo_mhz, pb_hi_mhz, alpha=0.07, color='green')
    ax_mag.axvline(-sb_lo_mhz, color='red', lw=0.8, ls=':', alpha=0.6)
    ax_mag.axvline( sb_lo_mhz, color='red', lw=0.8, ls=':', alpha=0.6,
                   label=f'Stopband edge (±{f_stop_mhz} MHz)')
    ax_mag.set_xlim(-fs_mhz / 2, fs_mhz / 2)
    ax_mag.set_xlabel('Frequency (MHz)', fontsize=FSIZE)
    ax_mag.set_ylabel('Magnitude (dB)', fontsize=FSIZE)
    ax_mag.tick_params(labelsize=FSIZE)
    ax_mag.grid(True, alpha=0.35)
    ax_mag.legend(fontsize=FSIZE - 2)

    # ---- row 2: in-band magnitude zoom ----------------------------------
    ax_zoom.plot(des_freqs_mhz, mag_des, lw=1.5, color=COLORS['des'],
                 label='Desired freq. resp.', zorder=1)
    ax_zoom.plot(w_cvx, mag_cvx, lw=1.8, color=COLORS['cvx'],
                 label='CVX fixed', zorder=3)
    ax_zoom.plot(w_rmz, mag_rmz, lw=1.8, color=COLORS['rmz'],
                 label='Remez IRLS', zorder=4)
    ax_zoom.axvspan(pb_lo_mhz, pb_hi_mhz, alpha=0.07, color='green')
    ax_zoom.axvline(-sb_lo_mhz, color='red', lw=0.8, ls=':', alpha=0.6)
    ax_zoom.axvline( sb_lo_mhz, color='red', lw=0.8, ls=':', alpha=0.6,
                    label=f'Stopband edge (±{f_stop_mhz} MHz)')
    ax_zoom.set_xlim(pb_lo_mhz - zoom_margin, pb_hi_mhz + zoom_margin)
    ax_zoom.set_ylim(-1, 0.2)
    ax_zoom.set_xlabel('Frequency (MHz)', fontsize=FSIZE)
    ax_zoom.set_ylabel('Magnitude (dB)', fontsize=FSIZE)
    ax_zoom.set_title('In-band zoom', fontsize=FSIZE)
    ax_zoom.tick_params(labelsize=FSIZE)
    ax_zoom.grid(True, alpha=0.35)
    ax_zoom.legend(fontsize=FSIZE - 2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png_path = os.path.join(out_dir, 'compare_lowpass_methods.png')
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved PNG  → {png_path}")

    # ====================================================================
    # Plotly interactive figure
    #   row 1 – magnitude (full-band)
    #   row 2 – in-band magnitude zoom (y range −10 to +10 dB)
    # ====================================================================
    pfig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Magnitude', 'In-band zoom'),
        vertical_spacing=0.14,
    )

    for row in (1, 2):
        pfig.add_trace(go.Scatter(
            x=des_freqs_mhz.tolist(), y=mag_des.tolist(),
            name='Desired freq. resp.', line=dict(color=COLORS['des'], width=1.5),
            legendgroup='des', showlegend=(row == 1),
        ), row=row, col=1)
        pfig.add_trace(go.Scatter(
            x=w_cvx.tolist(), y=mag_cvx.tolist(),
            name='CVX fixed', line=dict(color=COLORS['cvx'], width=2),
            legendgroup='cvx', showlegend=(row == 1),
        ), row=row, col=1)
        pfig.add_trace(go.Scatter(
            x=w_rmz.tolist(), y=mag_rmz.tolist(),
            name='Remez IRLS', line=dict(color=COLORS['rmz'], width=2),
            legendgroup='rmz', showlegend=(row == 1),
        ), row=row, col=1)
        pfig.add_vrect(x0=pb_lo_mhz, x1=pb_hi_mhz,
                       fillcolor='green', opacity=0.07, line_width=0,
                       row=row, col=1)
        for xval in [-sb_lo_mhz, sb_lo_mhz]:
            pfig.add_vline(x=xval, line=dict(color='red', width=1, dash='dot'),
                           row=row, col=1)

    pfig.update_xaxes(range=[pb_lo_mhz - zoom_margin, pb_hi_mhz + zoom_margin], row=2, col=1)

    FS = 14
    pfig.update_xaxes(title_text='Frequency (MHz)', title_font=dict(size=FS),
                      tickfont=dict(size=FS))
    pfig.update_yaxes(title_font=dict(size=FS), tickfont=dict(size=FS))
    pfig.update_yaxes(title_text='Magnitude (dB)', row=1, col=1)
    pfig.update_yaxes(title_text='Magnitude (dB)', range=[-1, 0.2], row=2, col=1)
    pfig.update_layout(
        title=dict(
            text=(f'Desired vs CVX (fixed+TB) vs Remez — N={N}  '
                  f'±{f_pass_mhz} MHz passband  {f_stop_mhz} MHz stopband'),
            font=dict(size=FS),
        ),
        hovermode='x unified',
        font=dict(size=FS),
        height=750,
    )
    html_path = os.path.join(out_dir, 'compare_lowpass_methods.html')
    pfig.write_html(html_path)
    print(f"Saved HTML → {html_path}")

    return h_cvx, h_rmz


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Overlay desired response, CVX lowpass, and Remez lowpass.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--N',             type=int,   default=63)
    p.add_argument('--fs',            type=float, default=750.0)
    p.add_argument('--f-pass',        type=float, default=162.5)
    p.add_argument('--f-stop',        type=float, default=250.0)
    p.add_argument('--delta-p',       type=float, default=0.05)
    p.add_argument('--delta-s',       type=float, default=0.10)
    p.add_argument('--lambda-reg',    type=float, default=1e-3)
    p.add_argument('--n-grid',        type=int,   default=512)
    p.add_argument('--n-iter-remez',  type=int,   default=150)
    p.add_argument('--solver-cvx',    type=str,   default=None,
                   help='CVXPY solver (CLARABEL, SCS, …); default: auto')
    p.add_argument('--out-dir',       type=str,   default=None)
    p.add_argument('--verbose',       action='store_true')
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()
    run_comparison(
        N=args.N,
        fs_mhz=args.fs,
        f_pass_mhz=args.f_pass,
        f_stop_mhz=args.f_stop,
        delta_p=args.delta_p,
        delta_s=args.delta_s,
        lambda_reg=args.lambda_reg,
        n_grid=args.n_grid,
        n_iter_remez=args.n_iter_remez,
        solver_cvx=args.solver_cvx,
        out_dir=args.out_dir,
        verbose=args.verbose,
    )

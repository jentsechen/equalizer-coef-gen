"""
RX equalizer FIR design entry point.

Steps
-----
1. Load measured channel response, compute equalizer target (1/channel) with a
   linear-phase trend, save the plots and the response as a .npy.
2. Run one complex-Remez FIR design pass (RemezConfig loaded from
   remez_config.json if present, otherwise dataclass defaults) and save
   evaluation plots.

Run from the repository root:
    python3 gen_rx_filter/run_design.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "trx_board_meas"))

from s2p import load_freq_resp_as_fft
from run_eqz_lowpass_remez import RemezConfig, design_eqz_lowpass_remez

_FIGURE_DIR = os.path.join(_HERE, "figure", "freq_resp")
_CONFIG_PATH = os.path.join(_HERE, "remez_config.json")


# ---------------------------------------------------------------------------
# Freq-response plot helpers
# ---------------------------------------------------------------------------


def _traces(resp_fft: np.ndarray, freq_mhz: np.ndarray, name: str):
    shifted = np.fft.fftshift(resp_fft)
    mag_db = 20 * np.log10(np.maximum(np.abs(shifted), 1e-12))
    phase_deg = np.degrees(np.unwrap(np.angle(shifted)))
    return (
        go.Scatter(x=freq_mhz.tolist(), y=mag_db.tolist(), name=name),
        go.Scatter(x=freq_mhz.tolist(), y=phase_deg.tolist(), name=name, showlegend=False),
    )


def _apply_layout(fig):
    fig.update_xaxes(title_text="Frequency (MHz)")
    fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
    fig.update_yaxes(title_text="Phase (deg)", row=2, col=1)
    fig.update_layout(font=dict(size=20), margin=dict(l=80, r=20, t=20, b=60))


def _save_overlay_fig(resps: list, freq_mhz: np.ndarray, out_path: str):
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.08)
    for resp, name in resps:
        mag_trace, phase_trace = _traces(resp, freq_mhz, name)
        fig.add_trace(mag_trace, row=1, col=1)
        fig.add_trace(phase_trace, row=2, col=1)
    _apply_layout(fig)
    fig.write_html(out_path)
    print(f"saved → {out_path}")

    fig_mpl, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), tight_layout=True)
    for resp, name in resps:
        shifted = np.fft.fftshift(resp)
        mag_db = 20 * np.log10(np.maximum(np.abs(shifted), 1e-12))
        phase_deg = np.degrees(np.unwrap(np.angle(shifted)))
        ax1.plot(freq_mhz, mag_db, label=name)
        ax2.plot(freq_mhz, phase_deg)
    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_ylabel("Phase (deg)")
    ax1.grid(True)
    ax2.grid(True)
    ax1.legend()
    png_path = os.path.splitext(out_path)[0] + ".png"
    fig_mpl.savefig(png_path, dpi=150)
    plt.close(fig_mpl)
    print(f"saved → {png_path}")


def _save_fig(resp_fft: np.ndarray, freq_mhz: np.ndarray, name: str, out_path: str):
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.08)
    mag_trace, phase_trace = _traces(resp_fft, freq_mhz, name)
    fig.add_trace(mag_trace, row=1, col=1)
    fig.add_trace(phase_trace, row=2, col=1)
    _apply_layout(fig)
    fig.write_html(out_path)
    print(f"saved → {out_path}")

    shifted = np.fft.fftshift(resp_fft)
    mag_db = 20 * np.log10(np.maximum(np.abs(shifted), 1e-12))
    phase_deg = np.degrees(np.unwrap(np.angle(shifted)))
    fig_mpl, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), tight_layout=True)
    ax1.plot(freq_mhz, mag_db)
    ax2.plot(freq_mhz, phase_deg)
    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_ylabel("Phase (deg)")
    ax1.grid(True)
    ax2.grid(True)
    png_path = os.path.splitext(out_path)[0] + ".png"
    fig_mpl.savefig(png_path, dpi=150)
    plt.close(fig_mpl)
    print(f"saved → {png_path}")


# ---------------------------------------------------------------------------
# Step 1: build and save equalizer final response
# ---------------------------------------------------------------------------


def build_eqz_final(
    ch_resp_path: str,
    n_fft: int = 256,
    fs_mhz: float = 750.0,
    delay_samples: int = 31,
    out_dir: str = _FIGURE_DIR,
) -> str:
    """Compute equalizer + linear-phase trend from a channel measurement.

    Saves a magnitude/phase HTML plot and a .npy of the final response.

    Args:
        ch_resp_path: Path to the raw channel measurement .npy file.
        delay_samples: Linear-phase delay (samples) added so the FIR target
            is causal (center tap index of a length-N FIR, e.g. (N-1)//2).
        out_dir: Directory where plots and the .npy are written.

    Returns:
        Path to the saved eqz_final .npy file.
    """
    ch_resp = load_freq_resp_as_fft(ch_resp_path, n_fft=n_fft, fs_mhz=fs_mhz)
    eqz_resp = 1.0 / ch_resp
    eqz_resp = eqz_resp / np.abs(eqz_resp[0])  # DC-normalise

    bin_freqs_rad = 2 * np.pi * np.fft.fftfreq(n_fft)
    eqz_final = eqz_resp * np.exp(-1j * bin_freqs_rad * delay_samples)

    freq_mhz = np.fft.fftshift(np.fft.fftfreq(n_fft)) * fs_mhz
    stem = os.path.splitext(os.path.basename(ch_resp_path))[0]
    if stem.endswith("_eqz"):
        stem = stem[:-4]

    os.makedirs(out_dir, exist_ok=True)
    _save_overlay_fig(
        [(ch_resp, "channel"), (eqz_resp, "equalizer")],
        freq_mhz,
        os.path.join(out_dir, f"{stem}_ch_eqz.html"),
    )
    _save_fig(
        eqz_final,
        freq_mhz,
        "equalizer + linear trend",
        os.path.join(out_dir, f"{stem}_eqz_final.html"),
    )

    npy_path = os.path.join(out_dir, f"{stem}_eqz_final.npy")
    np.save(npy_path, eqz_final)
    print(f"saved → {npy_path}")
    return npy_path, eqz_final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cfg = RemezConfig.from_json(_CONFIG_PATH) if os.path.exists(_CONFIG_PATH) else RemezConfig()

    print("=" * 60)
    print("Step 1: build equalizer final response")
    print("=" * 60)
    _, eqz_final = build_eqz_final(cfg.eqz_resp_path, fs_mhz=cfg.fs_mhz)

    print()
    print("=" * 60)
    print("Step 2: complex-Remez FIR design")
    print("=" * 60)
    design_eqz_lowpass_remez(cfg, save=True, eqz_resp=eqz_final)


if __name__ == "__main__":
    main()

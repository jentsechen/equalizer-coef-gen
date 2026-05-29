from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

_DATA_DIR = Path(__file__).parent / "data"
_FIGURE_DIR = Path(__file__).parent / "figure"

# Touchstone S DB interleaved columns: S11(1,2) S21(3,4) S12(5,6) S22(7,8)
_S21 = (3, 4)
_S12 = (5, 6)

_DECIMATE = 20  # ~2500 pts/file from 50k raw samples


def fcx_read_s2p(
    file_path: str,
    gain_index: int = 3,
    phase_index: int = 4,
    f_min: float = 9.325,
    f_max: float = 9.975,
):
    """Read an S2P file and return the band-limited response plus an equalization filter.

    Returns: frequencies (GHz), gain_db, phase_deg (detrended), s_complex, W, W_smooth
    """
    data = np.loadtxt(file_path, comments=("#", "!"), usecols=(0, gain_index, phase_index))

    freq = data[:, 0] * 1e-9
    mask = (freq >= f_min) & (freq <= f_max)
    frequencies = freq[mask]
    gain_db = data[:, 1][mask]
    phase_deg = data[:, 2][mask]

    magnitude = 10 ** (gain_db / 20)
    phase_rad = np.deg2rad(phase_deg)

    p = np.polyfit(frequencies, np.unwrap(phase_rad), 1)
    phase_rad = np.unwrap(phase_rad) - np.polyval(p, frequencies)
    phase_deg = np.rad2deg(np.unwrap(phase_rad))
    s_complex = magnitude * np.exp(1j * phase_rad)

    eps = 1e-12
    W = (1 / (magnitude + eps)) * np.exp(-1j * phase_rad)
    W = W / (np.max(np.abs(W)) + eps) * 0.95

    n = len(W)
    win = min(101, n if n % 2 == 1 else n - 1)
    gain_smooth = savgol_filter(np.abs(W), win, 3)
    phase_smooth = savgol_filter(np.unwrap(np.angle(W)), win, 3)
    W_smooth = gain_smooth * np.exp(1j * phase_smooth)

    return (
        np.array(frequencies),
        np.array(gain_db),
        np.array(phase_deg),
        np.array(s_complex),
        W,
        W_smooth,
    )


def _col_indices(stem: str):
    """Return (gain_col, phase_col) for the active forward path of a given filename stem."""
    prefix = stem.split("_")[0]
    if prefix in ("tx", "stx"):
        return _S12
    if prefix in ("rx", "srx"):
        return _S21
    raise ValueError(f"Unknown prefix '{prefix}' in '{stem}'")


def _load_decimated(path: Path):
    """Load a full-range S2P file with decimation and linear phase removal."""
    gain_col, phase_col = _col_indices(path.stem)
    data = np.loadtxt(path, comments=("!", "#"), usecols=(0, gain_col, phase_col))
    freq = data[::_DECIMATE, 0] * 1e-9
    gain = data[::_DECIMATE, 1]
    phase_rad = np.unwrap(np.deg2rad(data[::_DECIMATE, 2]))
    phase_deg = np.rad2deg(phase_rad - np.polyval(np.polyfit(freq, phase_rad, 1), freq))
    return freq, gain, phase_deg



from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

_DATA_DIR = Path(__file__).parent / "data"
_FIGURE_DIR = Path(__file__).parent / "figure"
_FREQ_RESP_DIR = Path(__file__).parent / "freq_resp"

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

    phase_rad = np.unwrap(phase_rad)
    phase_deg = np.rad2deg(phase_rad)
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


def save_freq_resp(stem: str, out_dir: Path = None, **kwargs) -> Path:
    """Load an S2P file by stem and save the complex frequency response as .npy.

    The saved array has shape (2, N) complex128:
      row 0 — frequencies in GHz (real part only)
      row 1 — complex S21 (magnitude * exp(j*phase))
    """
    path = _DATA_DIR / f"{stem}.s2p"
    freq, _, _, s_complex, _, _ = fcx_read_s2p(str(path), **kwargs)
    ref_idx = np.argmin(np.abs(freq - 9.65))
    s_complex = s_complex / np.abs(s_complex[ref_idx])
    out_dir = Path(out_dir) if out_dir is not None else _FREQ_RESP_DIR
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{stem}.npy"
    np.save(out, np.array([freq.astype(complex), s_complex]))
    return out


def load_freq_resp(path) -> tuple:
    """Load a .npy frequency response file.

    Returns: frequencies (GHz, float), s_complex (complex128)
    """
    data = np.load(path)
    return data[0].real, data[1]


def load_freq_resp_as_fft(path, n_fft: int = 256, fs_mhz: float = 750.0, fc_ghz: float = 9.65) -> np.ndarray:
    """Load a .npy freq response and return an n_fft-bin complex array in FFT bin order.

    Maps measurement frequencies (GHz) to digital rad/sample centered at fc_ghz,
    then interpolates onto the n_fft FFT bin grid. Bins outside the measurement
    range are filled with the nearest edge value.
    """
    freq_ghz, s_complex = load_freq_resp(path)
    freq_rad = (freq_ghz - fc_ghz) * 1000.0 / fs_mhz * 2 * np.pi

    order = np.argsort(freq_rad)
    sf, ss = freq_rad[order], s_complex[order]

    re_fn = interp1d(sf, ss.real, kind="linear", bounds_error=False,
                     fill_value=(ss.real[0], ss.real[-1]))
    im_fn = interp1d(sf, ss.imag, kind="linear", bounds_error=False,
                     fill_value=(ss.imag[0], ss.imag[-1]))

    bin_freqs_rad = 2 * np.pi * np.fft.fftfreq(n_fft)
    return re_fn(bin_freqs_rad) + 1j * im_fn(bin_freqs_rad)


def _col_indices(stem: str):
    """Return (gain_col, phase_col) for the active forward path of a given filename stem."""
    prefix = stem.split("_")[0]
    if prefix in ("tx", "stx"):
        return _S12
    if prefix in ("rx", "srx"):
        return _S21
    raise ValueError(f"Unknown prefix '{prefix}' in '{stem}'")


def _load_decimated(path: Path):
    """Load a full-range S2P file with decimation. Phase is unwrapped and
    zero-referenced at the first point so it plots as a straight line for
    linear-phase paths (global detrending is avoided because noisy phase
    at the band edges distorts the polyfit baseline)."""
    gain_col, phase_col = _col_indices(path.stem)
    data = np.loadtxt(path, comments=("!", "#"), usecols=(0, gain_col, phase_col))
    freq = data[::_DECIMATE, 0] * 1e-9
    gain = data[::_DECIMATE, 1]
    phase_rad = np.unwrap(np.deg2rad(data[::_DECIMATE, 2]))
    phase_deg = np.rad2deg(phase_rad - phase_rad[0])
    return freq, gain, phase_deg



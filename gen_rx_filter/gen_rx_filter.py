import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from scipy.signal import resample_poly
from chisel3cr.common import qt, eSign, eMSB, eLSB
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from tx_equalizer_design import TxEqzDesByChirp
from enum import Enum, auto

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class FreqRespMode(Enum):
    ONLY_AAF, ONLY_EQZ, AAF_EQZ_DIRECT_COMB, AAF_EQZ_JOINT_OPT = (
        auto(),
        auto(),
        auto(),
        auto(),
    )


class DistortedSig(Enum):
    UCDC = "update_aaf_freq_resp/ucdc_distorted_sig.npy"
    DC = "update_aaf_freq_resp/dc_distorted_sig.npy"


def apply_qntz_cpx(qntz_format: qt, data: np.ndarray) -> np.ndarray:
    """Apply fixed-point quantization independently to the real and imaginary parts."""
    apply = np.vectorize(qntz_format.apply)
    return apply(data.real) + 1j * apply(data.imag)


def plot_freq_resp(aaf_freq_resp_qntz_list):
    """Plot magnitude (dB) and unwrapped phase of all quantized AAF frequency responses."""
    n_bins = len(aaf_freq_resp_qntz_list[0])
    freq_axis = np.linspace(-375, 375, n_bins)
    figure = make_subplots(rows=2, cols=1)
    for i, freq_resp_qntz in enumerate(aaf_freq_resp_qntz_list):
        shifted = np.fft.fftshift(freq_resp_qntz)
        mag_db_qntz = 10 * np.log10(np.maximum(np.abs(shifted), 1e-12))
        phase_deg = np.degrees(np.unwrap(np.angle(shifted)))
        figure.add_trace(
            go.Scatter(x=freq_axis, y=mag_db_qntz, name=f"filter {i}"), row=1, col=1
        )
        figure.add_trace(
            go.Scatter(x=freq_axis, y=phase_deg, name=f"filter {i}", showlegend=False),
            row=2,
            col=1,
        )
    figure.update_layout(
        xaxis=dict(title="frequency (MHz)"),
        # yaxis=dict(title="magnitude (dB)", range=[-25, 1]),
        yaxis=dict(title="magnitude (dB)"),
        xaxis2=dict(title="frequency (MHz)"),
        yaxis2=dict(title="phase (deg)"),
        font=dict(size=20),
        margin=dict(l=80, r=20, t=20, b=60),
    )
    fig_dir = os.path.join(_SCRIPT_DIR, "figure")
    os.makedirs(fig_dir, exist_ok=True)
    figure.write_html(os.path.join(fig_dir, "aaf_freq_resp.html"))


def plot_time_resp(aaf_freq_resp_qntz_list):
    """Plot the impulse response (IFFT of freq resp) of all quantized AAF filters."""
    figure = make_subplots(rows=1, cols=1)
    for i, freq_resp_qntz in enumerate(aaf_freq_resp_qntz_list):
        time_resp_qntz = np.fft.ifft(freq_resp_qntz).real
        figure.add_trace(go.Scatter(y=time_resp_qntz[0:63], name=f"filter {i}"))
    figure.update_layout(
        xaxis=dict(title="sample"),
        yaxis=dict(title="amplitude"),
        font=dict(size=20),
        margin=dict(l=80, r=20, t=20, b=60),
    )
    fig_dir = os.path.join(_SCRIPT_DIR, "figure")
    os.makedirs(fig_dir, exist_ok=True)
    figure.write_html(os.path.join(fig_dir, "aaf_time_resp.html"))


def save_freq_resp_table(aaf_freq_resp_qntz_list):
    """Save all AAF frequency response coefficients as s1.14 integers to .txt files."""
    aaf_freq_resp_table = np.array(aaf_freq_resp_qntz_list).flatten()
    real_int = (aaf_freq_resp_table.real * 2**14).astype(int)
    imag_int = (aaf_freq_resp_table.imag * 2**14).astype(int)
    out_dir = os.path.join(_SCRIPT_DIR, "update_aaf_freq_resp")
    with open(os.path.join(out_dir, "aaf_freq_resp_table_real.txt"), "w") as f:
        f.write(",".join(map(str, real_int)))
    with open(os.path.join(out_dir, "aaf_freq_resp_table_imag.txt"), "w") as f:
        f.write(",".join(map(str, imag_int)))


def gen_eqz_imp_resp(sig: DistortedSig = DistortedSig.DC, n_taps=9):
    """Generate the equalizer impulse response from the measured distorted signal file."""
    tx_eqz_des_by_chirp = TxEqzDesByChirp(fs_is_750mhz=True)
    eqz_imp_resp = tx_eqz_des_by_chirp.gen_coef(
        file_path=os.path.join(_SCRIPT_DIR, sig.value), resample_meas=False, n_taps=n_taps
    )
    return eqz_imp_resp


def gen_eqz_freq_resp(sig: DistortedSig = DistortedSig.DC, n_taps=9):
    """Return the 256-bin FFT of the equalizer impulse response."""
    eqz_imp_resp = gen_eqz_imp_resp(sig=sig, n_taps=n_taps)
    return np.fft.fft(a=np.array(eqz_imp_resp), n=256)


def _compute_rx_filter_list(
    mode: FreqRespMode = FreqRespMode.ONLY_AAF, sig: DistortedSig = DistortedSig.DC
):
    """Compute and return the 9 quantized RX filter frequency responses for the given mode."""
    qntz_format = qt(
        sign=eSign.Signed, int_bit=1, frac_bit=14, msb=eMSB.Sat, lsb=eLSB.Rnd
    )
    with open(
        os.path.join(_SCRIPT_DIR, "aaf_freq_resp/aaf_freq_resp_256.json"),
        "r",
        encoding="UTF-8",
    ) as f:
        aaf_freq_resp_j = json.load(f)
    eqz_resp = (
        gen_eqz_freq_resp(sig)
        if mode in (FreqRespMode.AAF_EQZ_DIRECT_COMB, FreqRespMode.ONLY_EQZ)
        else None
    )
    aaf_freq_resp_qntz_list = []
    for i in range(9):
        aaf_resp = np.array(aaf_freq_resp_j["re"][i]) + 1j * np.array(
            aaf_freq_resp_j["im"][i]
        )
        if mode == FreqRespMode.ONLY_AAF:
            freq_resp = aaf_resp
        elif mode == FreqRespMode.AAF_EQZ_DIRECT_COMB:
            freq_resp = aaf_resp * eqz_resp
        # elif mode == FreqRespMode.AAF_EQZ_JOINT_OPT:
        #     pass
        elif mode == FreqRespMode.ONLY_EQZ:
            freq_resp = eqz_resp
        else:
            raise ValueError(f"Mode {mode} is not supported!")
        freq_resp /= max([max(freq_resp.real), max(freq_resp.imag)])
        aaf_freq_resp_qntz_list.append(apply_qntz_cpx(qntz_format, freq_resp))
    return aaf_freq_resp_qntz_list


def gen_rx_filter(
    mode: FreqRespMode = FreqRespMode.ONLY_AAF, sig: DistortedSig = DistortedSig.DC
):
    """Build, quantize, plot, and save the RX filter table for the selected FreqRespMode."""
    aaf_freq_resp_qntz_list = _compute_rx_filter_list(mode, sig)
    plot_freq_resp(aaf_freq_resp_qntz_list)
    plot_time_resp(aaf_freq_resp_qntz_list)
    save_freq_resp_table(aaf_freq_resp_qntz_list)


def plot_sig(sig: DistortedSig = DistortedSig.DC):
    """Plot distorted signal and its gen_eqz-compensated version (real, imag, envelope)."""
    eqz_imp_resp = gen_eqz_imp_resp(sig=sig)
    distorted_sig = np.load(os.path.join(_SCRIPT_DIR, sig.value))
    compensated_signal = np.convolve(a=distorted_sig, v=eqz_imp_resp)
    figure = make_subplots(rows=3, cols=1)
    figure.add_trace(
        go.Scatter(y=distorted_sig.real, name="distorted signal, real part"),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(y=distorted_sig.imag, name="distorted signal, imag. part"),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(y=abs(distorted_sig), name="distorted signal, envelope"),
        row=3,
        col=1,
    )
    figure.add_trace(
        go.Scatter(y=compensated_signal.real, name="compensated signal, real part"),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(y=compensated_signal.imag, name="compensated signal, imag. part"),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(y=abs(compensated_signal), name="compensated signal, envelope"),
        row=3,
        col=1,
    )
    figure.update_layout(
        font=dict(size=20),
        margin=dict(l=80, r=20, t=20, b=60),
    )
    fig_dir = os.path.join(_SCRIPT_DIR, "figure")
    os.makedirs(fig_dir, exist_ok=True)
    figure.write_html(os.path.join(fig_dir, "signals.html"))


def _chirp_rate(sig, fs):
    """Return instantaneous frequency (MHz) and chirp rate (MHz/µs) via linear fit."""
    inst_freq_hz = np.diff(np.unwrap(np.angle(sig))) / (2 * np.pi) * fs
    slope_hz_per_sample = np.polyfit(np.arange(len(inst_freq_hz)), inst_freq_hz, 1)[0]
    chirp_rate_mhz_per_us = slope_hz_per_sample * fs / 1e12
    return inst_freq_hz, chirp_rate_mhz_per_us


def plot_resample():
    """Plot instantaneous frequency of the 750 MHz chirp vs. the 1.25 GHz chirp resampled to 750 MHz."""
    with open("./training_sig/sig_chirp_750mhz.json", "r", encoding="UTF-8") as f:
        sig_750mhz_j = json.load(f)
    sig_750mhz = sig_750mhz_j["re"] + 1j * np.array(sig_750mhz_j["im"])
    with open("./training_sig/sig_chirp_s0_15.json", "r", encoding="UTF-8") as f:
        sig_1250mhz_j = json.load(f)
    sig_1250mhz = sig_1250mhz_j["re"] + 1j * np.array(sig_1250mhz_j["im"])
    sig_750mhz_by_resample = resample_poly(sig_1250mhz, up=3, down=5)

    _, cr_1250 = _chirp_rate(sig_1250mhz, fs=1.25e9)
    inst_freq_750, cr_750 = _chirp_rate(sig_750mhz, fs=0.75e9)
    inst_freq_750_rs, cr_750_rs = _chirp_rate(sig_750mhz_by_resample, fs=0.75e9)

    print(f"1.25GHz, chirp rate = {cr_1250:.3f} MHz/µs")
    print(f"750MHz, chirp rate = {cr_750:.3f} MHz/µs")
    print(f"750MHz by resample, chirp rate = {cr_750_rs:.3f} MHz/µs")

    fs_750 = 0.75e9
    time_750_us = np.arange(len(inst_freq_750)) / fs_750 * 1e6
    time_750_rs_us = np.arange(len(inst_freq_750_rs)) / fs_750 * 1e6

    figure = make_subplots(rows=1, cols=1)
    figure.add_trace(
        go.Scatter(x=time_750_us, y=inst_freq_750 / 1e6, name="750MHz"), row=1, col=1
    )
    figure.add_trace(
        go.Scatter(
            x=time_750_rs_us, y=inst_freq_750_rs / 1e6, name="750MHz by resample"
        ),
        row=1,
        col=1,
    )
    figure.update_layout(
        font=dict(size=20),
        xaxis=dict(title="time (us)", range=[0, 3.28]),
        yaxis=dict(title="frequency (MHz)"),
    )
    fig_dir = os.path.join(_SCRIPT_DIR, "figure")
    os.makedirs(fig_dir, exist_ok=True)
    figure.write_html(os.path.join(fig_dir, "resample.html"))


if __name__ == "__main__":
    gen_rx_filter(mode=FreqRespMode.AAF_EQZ_DIRECT_COMB, sig=DistortedSig.DC)
    plot_sig(sig=DistortedSig.DC)
    # plot_resample()

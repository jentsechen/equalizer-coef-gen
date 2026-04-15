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
from enum import Enum

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class FreqRespMode(Enum):
    ONLY_AAF = "only-aaf"
    AAF_EQZ  = "aaf-eqz"
    ONLY_EQZ = "only-eqz"


class DistortedSig(Enum):
    UCDC = "update_aaf_freq_resp/ucdc_distorted_sig.npy"
    DC   = "update_aaf_freq_resp/dc_distorted_sig.npy"


def apply_qntz_cpx(qntz_format: qt, data: np.ndarray) -> np.ndarray:
    apply = np.vectorize(qntz_format.apply)
    return apply(data.real) + 1j * apply(data.imag)


def plot_freq_resp(aaf_freq_resp_qntz_list):
    n_bins = len(aaf_freq_resp_qntz_list[0])
    freq_axis = np.linspace(-375, 375, n_bins)
    figure = make_subplots(rows=1, cols=1)
    for i, freq_resp_qntz in enumerate(aaf_freq_resp_qntz_list):
        mag_db_qntz = 10 * np.log10(
            np.maximum(np.abs(np.fft.fftshift(freq_resp_qntz)), 1e-12)
        )
        figure.add_trace(go.Scatter(x=freq_axis, y=mag_db_qntz, name=f"filter {i}"))
    figure.update_layout(
        xaxis=dict(title="frequency (MHz)"),
        # yaxis=dict(title="magnitude (dB)", range=[-25, 1]),
        yaxis=dict(title="magnitude (dB)"),
        font=dict(size=20),
        margin=dict(l=80, r=20, t=20, b=60),
    )
    fig_dir = os.path.join(_SCRIPT_DIR, "figure")
    os.makedirs(fig_dir, exist_ok=True)
    figure.write_html(os.path.join(fig_dir, "aaf_freq_resp.html"))


def plot_time_resp(aaf_freq_resp_qntz_list):
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
    aaf_freq_resp_table = np.array(aaf_freq_resp_qntz_list).flatten()
    real_int = (aaf_freq_resp_table.real * 2**14).astype(int)
    imag_int = (aaf_freq_resp_table.imag * 2**14).astype(int)
    out_dir = os.path.join(_SCRIPT_DIR, "update_aaf_freq_resp")
    with open(os.path.join(out_dir, "aaf_freq_resp_table_real.txt"), "w") as f:
        f.write(",".join(map(str, real_int)))
    with open(os.path.join(out_dir, "aaf_freq_resp_table_imag.txt"), "w") as f:
        f.write(",".join(map(str, imag_int)))


def eqz_freq_resp():
    file_name = "update_aaf_freq_resp/ucdc_distorted_sig.npy"
    # file_name = "update_aaf_freq_resp/dc_distorted_sig.npy"
    tx_eqz_des_by_chirp = TxEqzDesByChirp()
    eqz_imp_resp = tx_eqz_des_by_chirp.gen_coef(
        file_path=os.path.join(_SCRIPT_DIR, file_name)
    )
    return np.fft.fft(a=np.array(eqz_imp_resp), n=256)


def gen_rx_filter(mode: FreqRespMode = FreqRespMode.ONLY_AAF):
    qntz_format = qt(
        sign=eSign.Signed, int_bit=1, frac_bit=14, msb=eMSB.Sat, lsb=eLSB.Rnd
    )
    with open(
        os.path.join(_SCRIPT_DIR, "aaf_freq_resp/aaf_freq_resp_256.json"),
        "r",
        encoding="UTF-8",
    ) as f:
        aaf_freq_resp_j = json.load(f)
    eqz_resp = eqz_freq_resp() if mode in (FreqRespMode.AAF_EQZ, FreqRespMode.ONLY_EQZ) else None
    aaf_freq_resp_qntz_list = []
    for i in range(9):
        aaf_resp = np.array(aaf_freq_resp_j["re"][i]) + 1j * np.array(
            aaf_freq_resp_j["im"][i]
        )
        if mode == FreqRespMode.ONLY_AAF:
            freq_resp = aaf_resp
        elif mode == FreqRespMode.AAF_EQZ:
            freq_resp = aaf_resp * eqz_resp
        else:  # ONLY_EQZ
            freq_resp = eqz_resp
        aaf_freq_resp_qntz_list.append(apply_qntz_cpx(qntz_format, freq_resp))

    plot_freq_resp(aaf_freq_resp_qntz_list)
    plot_time_resp(aaf_freq_resp_qntz_list)
    save_freq_resp_table(aaf_freq_resp_qntz_list)


def plot_sig(sig: DistortedSig = DistortedSig.UCDC):
    file_path = os.path.join(_SCRIPT_DIR, sig.value)
    tx_eqz_des_by_chirp = TxEqzDesByChirp(resample_desired=True)
    eqz_imp_resp = tx_eqz_des_by_chirp.gen_coef(
        file_path=file_path, resample_meas=False
    )
    distorted_sig = np.load(file_path)
    compensated_signal = np.convolve(a=distorted_sig, v=eqz_imp_resp)
    figure = make_subplots(rows=3, cols=1)
    figure.add_trace(go.Scatter(y=distorted_sig.real, name="distorted signal, real part"), row=1, col=1)
    figure.add_trace(go.Scatter(y=distorted_sig.imag, name="distorted signal, imag. part"), row=2, col=1)
    figure.add_trace(go.Scatter(y=abs(distorted_sig), name="distorted signal, envelope"), row=3, col=1)
    figure.add_trace(go.Scatter(y=compensated_signal.real, name="compensated signal, real part"), row=1, col=1)
    figure.add_trace(go.Scatter(y=compensated_signal.imag, name="compensated signal, imag. part"), row=2, col=1)
    figure.add_trace(go.Scatter(y=abs(compensated_signal), name="compensated signal, envelope"), row=3, col=1)
    figure.update_layout(
        font=dict(size=20),
        margin=dict(l=80, r=20, t=20, b=60),
    )
    fig_dir = os.path.join(_SCRIPT_DIR, "figure")
    os.makedirs(fig_dir, exist_ok=True)
    figure.write_html(os.path.join(fig_dir, "signals.html"))


if __name__ == "__main__":
    gen_rx_filter(mode=FreqRespMode.AAF_EQZ)
    plot_sig(sig=DistortedSig.DC)

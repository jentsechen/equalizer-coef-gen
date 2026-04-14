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

def apply_qntz_cpx(qntz_format: qt, data: np.ndarray) -> np.ndarray:
    apply = np.vectorize(qntz_format.apply)
    return apply(data.real) + 1j * apply(data.imag)

def plot_freq_resp(aaf_freq_resp_qntz_list):
    n_bins = len(aaf_freq_resp_qntz_list[0])
    freq_axis = np.linspace(-375, 375, n_bins)
    figure = make_subplots(rows=1, cols=1)
    for i, freq_resp_qntz in enumerate(aaf_freq_resp_qntz_list):
        mag_db_qntz = 10 * np.log10(np.maximum(np.abs(np.fft.fftshift(freq_resp_qntz)), 1e-12))
        figure.add_trace(go.Scatter(x=freq_axis, y=mag_db_qntz, name=f"filter {i}"))
    figure.update_layout(
        xaxis=dict(title="frequency (MHz)"),
        # yaxis=dict(title="magnitude (dB)", range=[-25, 1]),
        yaxis=dict(title="magnitude (dB)"),
        font=dict(size=20),
        margin=dict(l=80, r=20, t=20, b=60)
    )
    os.makedirs("figure", exist_ok=True)
    figure.write_html("figure/aaf_freq_resp.html")
    figure.write_image("figure/aaf_freq_resp.png")

def plot_ch_imp_resp():
    ch_imp_resp = np.load("ch_imp_resp.npy")
    figure = make_subplots(rows=1, cols=1)
    figure.add_trace(go.Scatter(y=ch_imp_resp.real, name="real"))
    figure.add_trace(go.Scatter(y=ch_imp_resp.imag, name="imag"))
    figure.update_layout(
        xaxis=dict(title="sample"),
        yaxis=dict(title="amplitude"),
        font=dict(size=20),
        margin=dict(l=80, r=20, t=20, b=60)
    )
    os.makedirs("figure", exist_ok=True)
    figure.write_html("figure/ch_imp_resp.html")
    figure.write_image("figure/ch_imp_resp.png")

def plot_distorted_sig():
    distorted_sig = np.load("distorted_sig.npy")
    figure = make_subplots(rows=1, cols=1)
    figure.add_trace(go.Scatter(y=distorted_sig.real, name="real"))
    figure.add_trace(go.Scatter(y=distorted_sig.imag, name="imag"))
    figure.update_layout(
        xaxis=dict(title="sample"),
        yaxis=dict(title="amplitude"),
        font=dict(size=20),
        margin=dict(l=80, r=20, t=20, b=60)
    )
    os.makedirs("figure", exist_ok=True)
    figure.write_html("figure/distorted_sig.html")
    figure.write_image("figure/distorted_sig.png")

def plot_time_resp(aaf_freq_resp_qntz_list):
    figure = make_subplots(rows=1, cols=1)
    for i, freq_resp_qntz in enumerate(aaf_freq_resp_qntz_list):
        time_resp_qntz = np.fft.ifft(freq_resp_qntz).real
        figure.add_trace(go.Scatter(y=time_resp_qntz[0:63], name=f"filter {i}"))
    figure.update_layout(
        xaxis=dict(title="sample"),
        yaxis=dict(title="amplitude"),
        font=dict(size=20),
        margin=dict(l=80, r=20, t=20, b=60)
    )
    os.makedirs("figure", exist_ok=True)
    figure.write_html("figure/aaf_time_resp.html")
    figure.write_image("figure/aaf_time_resp.png")

def save_freq_resp_table(aaf_freq_resp_qntz_list):
    aaf_freq_resp_table = np.array(aaf_freq_resp_qntz_list).flatten()
    real_int = (aaf_freq_resp_table.real * 2**14).astype(int)
    imag_int = (aaf_freq_resp_table.imag * 2**14).astype(int)
    with open("update_aaf_freq_resp/aaf_freq_resp_table_real.txt", "w") as f:
        f.write(",".join(map(str, real_int)))
    with open("update_aaf_freq_resp/aaf_freq_resp_table_imag.txt", "w") as f:
        f.write(",".join(map(str, imag_int)))

def eqz_freq_resp():
    # eqz_imp_resp = np.load("update_aaf_freq_resp/eqz_imp_resp.npy")
    # eqz_imp_resp_resampled = resample_poly(eqz_imp_resp, up=3, down=5)
    # eqz_imp_resp = np.load("ch_imp_resp.npy")[196:240]
    # eqz_imp_resp /= sum(eqz_imp_resp)
    eqz_imp_resp = [ 
        0.0328235 -0.01797049j, -0.09734622+0.05522723j,  0.17209393-0.10148092j,
        -0.22323889+0.14327868j,  0.23938844-0.16465379j, -0.22117735+0.13983384j,
        0.16539116-0.07544018j, -0.08902833+0.01685696j,  0.02728992+0.00434867j]
    return np.fft.fft(a=np.array(eqz_imp_resp), n=256)

def main(enable_eqz: bool = False):
    qntz_format = qt(sign=eSign.Signed, int_bit=1, frac_bit=14, msb=eMSB.Sat, lsb=eLSB.Rnd)
    with open("aaf_freq_resp/aaf_freq_resp_256.json", "r", encoding="UTF-8") as f:
        aaf_freq_resp_j = json.load(f)
    eqz_resp = eqz_freq_resp() if enable_eqz else None
    aaf_freq_resp_qntz_list = []
    for i in range(9):
        freq_resp = np.array(aaf_freq_resp_j["re"][i]) + 1j*np.array(aaf_freq_resp_j["im"][i])
        if enable_eqz:
            freq_resp = freq_resp * eqz_resp
            # freq_resp = freq_resp
            # freq_resp = eqz_resp
            # freq_resp = freq_resp / eqz_resp
        aaf_freq_resp_qntz_list.append(apply_qntz_cpx(qntz_format, freq_resp))

    plot_freq_resp(aaf_freq_resp_qntz_list)
    plot_time_resp(aaf_freq_resp_qntz_list)
    save_freq_resp_table(aaf_freq_resp_qntz_list)

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--enable-eqz", action="store_true", help="Multiply AAF freq response with equalizer freq response")
    # args = parser.parse_args()
    # main(enable_eqz=args.enable_eqz)
    # plot_distorted_sig()
    tx_eqz_des_by_chirp = TxEqzDesByChirp()
    eqz_imp_resp = tx_eqz_des_by_chirp.gen_coef(file_path="distorted_sig.npy")
    # plot_ch_imp_resp()

    distorted_sig = np.load("distorted_sig.npy")
    compensated_signal = np.convolve(a=distorted_sig, v=eqz_imp_resp)
    figure = make_subplots(rows=3, cols=1)
    figure.add_trace(go.Scatter(y=distorted_sig.real, name="real"), row=1, col=1)
    figure.add_trace(go.Scatter(y=distorted_sig.imag, name="imag"), row=2, col=1)
    figure.add_trace(go.Scatter(y=abs(distorted_sig)), row=3, col=1)
    figure.add_trace(go.Scatter(y=compensated_signal.real, name="real"), row=1, col=1)
    figure.add_trace(go.Scatter(y=compensated_signal.imag, name="imag"), row=2, col=1)
    figure.add_trace(go.Scatter(y=abs(compensated_signal)), row=3, col=1)
    figure.update_layout(
        xaxis=dict(title="sample"),
        font=dict(size=20),
        margin=dict(l=80, r=20, t=20, b=60)
    )
    os.makedirs("figure", exist_ok=True)
    figure.write_html("figure/signals.html")
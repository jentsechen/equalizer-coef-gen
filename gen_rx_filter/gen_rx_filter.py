import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from chisel3cr.common import qt, eSign, eMSB, eLSB
import plotly.graph_objs as go
from plotly.subplots import make_subplots

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
        yaxis=dict(title="magnitude (dB)", range=[-25, 1]),
        font=dict(size=20),
        margin=dict(l=80, r=20, t=20, b=60)
    )
    figure.write_html("aaf_freq_resp.html")
    figure.write_image("aaf_freq_resp.png")

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
    figure.write_html("aaf_time_resp.html")
    figure.write_image("aaf_time_resp.png")

def main():
    qntz_format = qt(sign=eSign.Signed, int_bit=1, frac_bit=14, msb=eMSB.Sat, lsb=eLSB.Rnd)
    with open("aaf_freq_resp_256.json", "r", encoding="UTF-8") as f:
        aaf_freq_resp_j = json.load(f)
    aaf_freq_resp_qntz_list = []
    for i in range(9):
        freq_resp = np.array(aaf_freq_resp_j["re"][i]) + 1j*np.array(aaf_freq_resp_j["im"][i])
        aaf_freq_resp_qntz_list.append(apply_qntz_cpx(qntz_format, freq_resp))

    plot_freq_resp(aaf_freq_resp_qntz_list)
    plot_time_resp(aaf_freq_resp_qntz_list)

if __name__ == "__main__":
    main()
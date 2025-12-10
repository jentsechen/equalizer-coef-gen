import struct
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pof
from plotly.subplots import make_subplots
import json
from tx_equalizer_design import TxEqzDesByChirp
from scipy.signal import resample_poly
import matplotlib.pyplot as plt

def load_bin_data(file_name):
    with open("./{}.bin".format(file_name), "rb") as f:
        data = f.read()
    n_bytes = len(data)
    decode_data = []
    for i in range(int(n_bytes/32)):
        for j in range(5):
            s = i*32 + j*4
            # re = float(struct.unpack("h", data[(s+2):(s+4)])[0])
            # im = float(struct.unpack("h", data[s:(s+2)])[0])
            re = float(struct.unpack("h", data[s:(s+2)])[0])
            im = float(struct.unpack("h", data[(s+2):(s+4)])[0])
            decode_data.append((re + 1j*im)/2**14)
    return np.array(decode_data)

def save_bin_data(data, file_name):
    assert len(data) % 5 == 0
    depth = len(data) // 5
    interleaved = np.empty(int(depth * 2 * 8), dtype=np.float32)
    for i in range(depth):
        for j in range(5):
            interleaved[i*16+j*2] = data[i*5+j].real
            interleaved[i*16+j*2+1] = data[i*5+j].imag
            # interleaved[1::2] = data.imag
            # interleaved[0::2] = data.real
    qntz_data = np.clip(np.round(interleaved * 2**14), -32768, 32767).astype(np.int16)
    with open("./{}.bin".format(file_name), "wb") as f:
        qntz_data.tofile(f)

def load_bin_data(file_name):
    with open("./{}.bin".format(file_name), "rb") as f:
        data = f.read()
    n_bytes = len(data)
    decode_data = []
    for i in range(int(n_bytes/32)):
        for j in range(5):
            s = i*32 + j*4
            # re = float(struct.unpack("h", data[(s+2):(s+4)])[0])
            # im = float(struct.unpack("h", data[s:(s+2)])[0])
            re = float(struct.unpack("h", data[s:(s+2)])[0])
            im = float(struct.unpack("h", data[(s+2):(s+4)])[0])
            decode_data.append((re + 1j*im)/2**14)
    return np.array(decode_data)

if __name__ == "__main__":
    tx_eqz_des_by_chirp = TxEqzDesByChirp()
    coef_ch4 = tx_eqz_des_by_chirp.gen_coef(file_path="./meas_sig/result/ch4_bfr_eqz.npy", print_en=True)
    coef_ch5 = tx_eqz_des_by_chirp.gen_coef(file_path="./meas_sig/result/ch5_bfr_eqz.npy", print_en=True)

    n_sample = 20200
    ch_idx = 4
    time_dur = 20
    sig_before_eqz = np.load("./meas_sig/result/ch{}_bfr_eqz_{}us.npy".format(ch_idx, time_dur))
    sig_after_eqz = np.load("./meas_sig/result/ch{}_aft_eqz_{}us.npy".format(ch_idx, time_dur))
    figure = make_subplots(rows=2, cols=1)
    figure.add_trace(go.Scatter(y=sig_before_eqz.real[0:n_sample], name="not equalized, real part"), row=1, col=1)
    figure.add_trace(go.Scatter(y=sig_before_eqz.imag[0:n_sample], name="not equalized, imaginary part"), row=2, col=1)
    figure.add_trace(go.Scatter(y=sig_after_eqz.real[0:n_sample], name="equalized, real part"), row=1, col=1)
    figure.add_trace(go.Scatter(y=sig_after_eqz.imag[0:n_sample], name="equalized, imaginary part"), row=2, col=1)
    figure.update_layout(
        xaxis=dict(title="sample"),
        xaxis2=dict(title="sample"),
        font=dict(size=25)
    )
    # pof.iplot(figure)

    ch_idx = 5
    signal_analyzer_output = np.load("./meas_sig/result/ch{}_bfr_eqz.npy".format(ch_idx))
    tx_eqz_des_by_chirp = TxEqzDesByChirp()
    unwanted_signal = tx_eqz_des_by_chirp.proc_meas_to_train(signal_analyzer_output[0:4000])
    if ch_idx == 4:
        coef = coef_ch4
    if ch_idx == 5:
        coef = coef_ch5
    equalized_signal = np.convolve(unwanted_signal, coef)[0:len(tx_eqz_des_by_chirp.desired_signal)]
    figure = make_subplots(rows=2, cols=1)
    figure.add_trace(go.Scatter(y=unwanted_signal.real), row=1, col=1)
    figure.add_trace(go.Scatter(y=unwanted_signal.imag), row=2, col=1)
    figure.add_trace(go.Scatter(y=equalized_signal.real), row=1, col=1)
    figure.add_trace(go.Scatter(y=equalized_signal.imag), row=2, col=1)
    unwanted_sig_env = abs(unwanted_signal)
    equalized_sig_env = abs(equalized_signal)
    pof.iplot(figure)
    print(10*np.log10(max(unwanted_sig_env) / max(equalized_sig_env)))

    # with open("gen_seq_fixed.json", "r", encoding="UTF-8") as f:
    #     seq_j = json.load(f)
    # seq = seq_j["re"] + 1j*np.array(seq_j["im"])
    # seq_freq = np.fft.fft(seq[0:1024])
    # figure = make_subplots(rows=3, cols=1)
    # figure.add_trace(go.Scatter(y=seq_j["re"], name="real part"), row=1, col=1)
    # figure.add_trace(go.Scatter(y=seq_j["im"], name="imaginary part"), row=2, col=1)
    # figure.add_trace(go.Scatter(y=abs(seq_freq)), row=3, col=1)
    # pof.iplot(figure)
    # save_bin_data(seq, "pow_on_cal_train_seq")
    # seq_l = load_bin_data("pow_on_cal_train_seq")
    # figure = make_subplots(rows=2, cols=1)
    # figure.add_trace(go.Scatter(y=seq_l.real, name="real part"), row=1, col=1)
    # figure.add_trace(go.Scatter(y=seq_l.imag, name="imaginary part"), row=2, col=1)
    # pof.iplot(figure)

    print("DONE")
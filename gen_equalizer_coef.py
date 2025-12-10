import struct
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pof
from plotly.subplots import make_subplots
import json
from tx_equalizer_design import TxEqzDesByChirp
from scipy.signal import resample_poly
import matplotlib.pyplot as plt

def print_arr(data, name):
    arr = "int16_t impulse_coef_" + name + "_s1_14[9] = {"
    for i in range(8):
        arr += "{}, ".format(data[i])
    arr += (str(data[8]) + "};")
    return arr

def gen_norm_factor(coef):
    with open("./training_sig/sig_chirp_s0_15.json", "r", encoding="UTF-8") as f:
        desired_signal_j = json.load(f)
    desired_signal = desired_signal_j["re"] + 1j*np.array(desired_signal_j["im"])
    equalized_signal = np.convolve(desired_signal, coef)
    norm_factor = max(max(abs(equalized_signal.real)), max(abs(equalized_signal.imag)))
    print(norm_factor)
    return np.ceil(norm_factor * 100) / 100

def gen_coef(ch_idx):
    signal_analyzer_output = np.load("./meas_sig/result/ch{}_bfr_eqz.npy".format(ch_idx))
    tx_eqz_des_by_chirp = TxEqzDesByChirp()
    unwanted_signal = tx_eqz_des_by_chirp.proc_meas_to_train(signal_analyzer_output[0:4000].imag+1j*signal_analyzer_output[0:4000].real)
    coef = tx_eqz_des_by_chirp.train_coef(unwanted_signal=unwanted_signal, n_taps=9)
    norm_factor = gen_norm_factor(coef)
    coef /= norm_factor
    print(norm_factor)
    print(print_arr(np.array(np.round(coef.real*2**14), dtype=int), "re"))
    print(print_arr(np.array(np.round(coef.imag*2**14), dtype=int), "im"))
    return coef

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
    with open("./training_sig/sig_chirp_s0_15.json", "r", encoding="UTF-8") as f:
        desired_signal_j = json.load(f)
    desired_signal = desired_signal_j["re"] + 1j*np.array(desired_signal_j["im"])
    capture_memory_output = np.load("./meas_sig/equa_step0_ch4_cap.npy")
    signal_analyzer_output = np.load("./meas_sig/equa_step0_ch4_sa.npy")
    signal_analyzer_output = np.load("./meas_sig/equa_step0_ch5_sa.npy")
    
    coef_ch4 = gen_coef(ch_idx=4)
    coef_ch5 = gen_coef(ch_idx=5)

    # # pulse_width_10us = np.load("./meas_sig/result/ch4_100us.npy")[0:10150]
    # # pulse_width_10us_equalized = np.load("./meas_sig/result/ch4_100us_v2.npy")[0:10150]
    # # pulse_width_10us = np.load("./meas_sig/result/ch5_200us.npy")[0:20300]
    # pulse_width_10us_equalized = np.load("./meas_sig/result/ch4_200us_v2.npy")[0:20300]
    # figure = make_subplots(rows=2, cols=1)
    # # figure.add_trace(go.Scatter(y=pulse_width_10us.real, name="not equalized, real part"), row=1, col=1)
    # # figure.add_trace(go.Scatter(y=pulse_width_10us.imag, name="not equalized, imaginary part"), row=2, col=1)
    # figure.add_trace(go.Scatter(y=pulse_width_10us_equalized.real, name="equalized, real part"), row=1, col=1)
    # figure.add_trace(go.Scatter(y=pulse_width_10us_equalized.imag, name="equalized, imaginary part"), row=2, col=1)
    # figure.update_layout(
    #     xaxis=dict(title="sample"),
    #     xaxis2=dict(title="sample"),
    #     font=dict(size=25)
    # )
    # pof.iplot(figure)

    # sig_before_eqz = np.load("./meas_sig/result/ch5_bfr_eqz.npy")
    # sig_after_eqz = np.load("./meas_sig/result/ch5_aft_eqz.npy")
    # figure = make_subplots(rows=2, cols=1)
    # figure.add_trace(go.Scatter(y=sig_before_eqz.real[0:3500], name="not equalized, real part"), row=1, col=1)
    # figure.add_trace(go.Scatter(y=sig_before_eqz.imag[0:3500], name="not equalized, imaginary part"), row=2, col=1)
    # figure.add_trace(go.Scatter(y=sig_after_eqz.real[0:3500], name="equalized, real part"), row=1, col=1)
    # figure.add_trace(go.Scatter(y=sig_after_eqz.imag[0:3500], name="equalized, imaginary part"), row=2, col=1)
    # figure.update_layout(
    #     xaxis=dict(title="sample"),
    #     xaxis2=dict(title="sample"),
    #     font=dict(size=25)
    # )
    # pof.iplot(figure)

    # n_sample = 20200
    # ch_idx = 5
    # time_dur = 20
    # sig_before_eqz = np.load("./meas_sig/result/ch{}_bfr_eqz_{}us.npy".format(ch_idx, time_dur))
    # sig_after_eqz = np.load("./meas_sig/result/ch{}_aft_eqz_{}us.npy".format(ch_idx, time_dur))
    # figure = make_subplots(rows=2, cols=1)
    # figure.add_trace(go.Scatter(y=sig_before_eqz.real[0:n_sample], name="not equalized, real part"), row=1, col=1)
    # figure.add_trace(go.Scatter(y=sig_before_eqz.imag[0:n_sample], name="not equalized, imaginary part"), row=2, col=1)
    # figure.add_trace(go.Scatter(y=sig_after_eqz.real[0:n_sample], name="equalized, real part"), row=1, col=1)
    # figure.add_trace(go.Scatter(y=sig_after_eqz.imag[0:n_sample], name="equalized, imaginary part"), row=2, col=1)
    # figure.update_layout(
    #     xaxis=dict(title="sample"),
    #     xaxis2=dict(title="sample"),
    #     font=dict(size=25)
    # )
    # pof.iplot(figure)

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

    n_sample = 4200
    ch_idx = 5
    sig_before_eqz = np.load("./meas_sig/ofdm/ch{}_bfr_eqz.npy".format(ch_idx))
    sig_after_eqz = np.load("./meas_sig/ofdm/ch{}_aft_eqz.npy".format(ch_idx))
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
    pof.iplot(figure)

    print("DONE")
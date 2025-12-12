import struct
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pof
from plotly.subplots import make_subplots
import json
from tx_equalizer_design import TxEqzDesByChirp
from scipy.signal import resample_poly
import matplotlib.pyplot as plt

def mag_resp(waveform):
    return abs(np.fft.fftshift(np.fft.fft(waveform)))

if __name__ == "__main__":
    ch_idx = 4
    file_path = "./meas_sig/result/ch{}_bfr_eqz.npy".format(ch_idx)
    tx_eqz_des_by_chirp = TxEqzDesByChirp()
    signal_analyzer_output = np.load(file_path)
    print(tx_eqz_des_by_chirp.find_pslr_db(tx_eqz_des_by_chirp.gen_meas_mf_out(signal_analyzer_output)))
    print(tx_eqz_des_by_chirp.find_irw_m(tx_eqz_des_by_chirp.gen_meas_mf_out(signal_analyzer_output)))
    print()

    coef = tx_eqz_des_by_chirp.gen_coef(file_path=file_path)
    unwanted_signal = tx_eqz_des_by_chirp.proc_meas_to_train(signal_analyzer_output[0:4000])
    equalized_signal = np.convolve(a=unwanted_signal, v=coef, mode='same')
    print(tx_eqz_des_by_chirp.find_pslr_db(tx_eqz_des_by_chirp.gen_mf_out(equalized_signal)))
    print(tx_eqz_des_by_chirp.find_irw_m(tx_eqz_des_by_chirp.gen_mf_out(equalized_signal)))
    print()

    figure = make_subplots(rows=2, cols=1)
    figure.add_trace(go.Scatter(y=unwanted_signal.real), row=1, col=1)
    figure.add_trace(go.Scatter(y=unwanted_signal.imag), row=2, col=1)
    figure.add_trace(go.Scatter(y=equalized_signal.real), row=1, col=1)
    figure.add_trace(go.Scatter(y=equalized_signal.imag), row=2, col=1)
    # pof.iplot(figure)

    figure = make_subplots(rows=1, cols=1)
    figure.add_trace(go.Scatter(y=mag_resp(unwanted_signal)), row=1, col=1)
    figure.add_trace(go.Scatter(y=mag_resp(equalized_signal)), row=1, col=1)
    pof.iplot(figure)

    print("DONE")
import numpy as np
import json
import plotly.graph_objs as go
import plotly.offline as pof
from plotly.subplots import make_subplots
from chisel3cr.common import qt, eSign, eMSB, eLSB
from tx_equalizer_design import TxEqzDesByChirp, TxEqzDesByOfdm
import struct
from scipy.signal import resample_poly
from scipy.signal import remez, freqz
from non_lin_effect_simulator import NonLinEffectSimulator, PowAmpMode

def sim_part_to_part_diff():
    n_test = 100
    pslr_sum, irw_sum = 0, 0
    for i in range(n_test):
        non_lin_effect_sim = NonLinEffectSimulator(n_rf_path=40, rf_path_p_to_p_diff_en=True)
        distorted_signal = non_lin_effect_sim.apply_channel(signal=non_lin_effect_sim.tx_eqz_des_by_chirp.desired_signal * 2**-7,
                                                        pow_amp_mode=PowAmpMode.NonLinRapp)
        pslr, irw = non_lin_effect_sim.find_perf_metric(distorted_signal)
        print(i, pslr, irw)
        pslr_sum += pslr
        irw_sum += irw
    print(pslr_sum/n_test, irw_sum/n_test)

if __name__ == "__main__":
    non_lin_effect_sim = NonLinEffectSimulator(n_rf_path=40, rf_path_p_to_p_diff_en=True)
    # distorted_signal = non_lin_effect_sim.apply_channel(signal=non_lin_effect_sim.tx_eqz_des_by_chirp.desired_signal,
    #                                                     pow_amp_mode=PowAmpMode.NonLinRapp)
    distorted_signal = non_lin_effect_sim.apply_channel(signal=non_lin_effect_sim.tx_eqz_des_by_chirp.desired_signal,
                                                        pow_amp_mode=PowAmpMode.NonLinLut)
    # distorted_signal = non_lin_effect_sim.apply_channel(signal=non_lin_effect_sim.tx_eqz_des_by_chirp.desired_signal,
    #                                                     pow_amp_mode=PowAmpMode.Linear)
    print(non_lin_effect_sim.find_perf_metric(non_lin_effect_sim.tx_eqz_des_by_chirp.desired_signal))
    print(non_lin_effect_sim.find_perf_metric(distorted_signal))

    ch4_chirp_meas_data_iter1 = non_lin_effect_sim.tx_eqz_des_by_chirp.proc_meas_to_train(np.load("./ota_meas_data/ota_out_iter_1/ch4_sig_chirp.npy"))
    print(non_lin_effect_sim.find_perf_metric(ch4_chirp_meas_data_iter1))

    # figure = make_subplots(rows=2, cols=1)
    # figure.add_trace(go.Scatter(y=ch4_chirp_meas_data_iter1.real, line=dict(width=2)), row=1, col=1)
    # figure.add_trace(go.Scatter(y=ch4_chirp_meas_data_iter1.imag, line=dict(width=2)), row=2, col=1)
    # figure.update_layout(
    #     xaxis=dict(title="sample"),       
    #     yaxis=dict(title="amplitude (real part)"),
    #     xaxis2=dict(title="sample"),      
    #     yaxis2=dict(title="amplitude (imaginary part)"),
    #     font=dict(size=20)
    # )
    # figure.write_html("measure_result.html")

    figure = make_subplots(rows=2, cols=1)
    figure.add_trace(go.Scatter(y=distorted_signal.real, line=dict(width=2)), row=1, col=1)
    figure.add_trace(go.Scatter(y=distorted_signal.imag, line=dict(width=2)), row=2, col=1)
    figure.update_layout(
        xaxis=dict(title="sample"),       
        yaxis=dict(title="amplitude (real part)"),
        xaxis2=dict(title="sample"),      
        yaxis2=dict(title="amplitude (imaginary part)"),
        font=dict(size=20)
    )
    figure.write_html("simulation_result.html")

    figure = make_subplots(rows=1, cols=1)
    for i in range(non_lin_effect_sim.n_rf_path):
        rf_path_imp_resp = np.concatenate([non_lin_effect_sim.rf_path_imp_resp_list[i], np.zeros(len(non_lin_effect_sim.rf_path_imp_resp_list[i])*10)])
        rf_path_freq_resp = np.fft.fftshift(np.fft.fft(rf_path_imp_resp))
        figure.add_trace(go.Scatter(x=np.linspace(-1.25, 1.25, len(rf_path_freq_resp)),
                                    y=20*np.log10(abs(rf_path_freq_resp))), row=1, col=1)
    figure.update_layout(xaxis=dict(title="frequency (GHz)"), yaxis=dict(title="magnitude (dB)"), font=dict(size=20))
    figure.write_html("rf_path_freq_resp.html")
    
    print("DONE")


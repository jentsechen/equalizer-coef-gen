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
from scipy.signal import butter, lfilter, freqz, cheby1

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

def plot_filter_freq_resp(b, a):
    w, h = freqz(b, a, fs=1.25e9)
    figure = make_subplots(rows=2, cols=1)
    figure.add_trace(go.Scatter(x=w/1e6, y=20*np.log10(abs(h))), row=1, col=1)
    figure.add_trace(go.Scatter(x=w/1e6, y=np.unwrap(np.angle(h)) * 180 / np.pi), row=2, col=1)
    figure.update_yaxes(range=[-10, 5], row=1, col=1)
    figure.update_layout(
        xaxis=dict(title="Frequency (MHz)"),
        yaxis=dict(title="Magnitude (dB)"),
        xaxis2=dict(title="Frequency (MHz)"),
        yaxis2=dict(title="Phase (degree)"),
        font=dict(size=20)
    )
    figure.write_html("filter_freq_resp.html")

if __name__ == "__main__":
    non_lin_effect_sim = NonLinEffectSimulator(n_rf_path=40, rf_path_p_to_p_diff_en=True, rand_cpx_gain_en=True)

    n_taps = 31
    pow_amp_mode = PowAmpMode.NonLinLut
    # pow_amp_mode = PowAmpMode.Linear
    tx_eqz_des = non_lin_effect_sim.tx_eqz_des_by_chirp
    desired_signal = tx_eqz_des.desired_signal
    distorted_signal_0 = non_lin_effect_sim.apply_channel(signal=desired_signal, pow_amp_mode=pow_amp_mode)

    coef_0 = tx_eqz_des.train_coef(unwanted_signal=distorted_signal_0, n_taps=n_taps)
    eqz_out_0 = np.convolve(desired_signal, coef_0.conjugate(), mode='full')[0:len(desired_signal)]
    distorted_signal_1 = non_lin_effect_sim.apply_channel(signal=eqz_out_0, pow_amp_mode=pow_amp_mode)

    coef_1 = tx_eqz_des.train_coef(unwanted_signal=distorted_signal_1, n_taps=n_taps, desired_signal=eqz_out_0)
    eqz_out_1 = np.convolve(desired_signal, coef_1.conjugate(), mode='full')[0:len(desired_signal)]
    distorted_signal_2 = non_lin_effect_sim.apply_channel(signal=eqz_out_1, pow_amp_mode=pow_amp_mode)

    coef_2 = tx_eqz_des.train_coef(unwanted_signal=distorted_signal_2, n_taps=n_taps, desired_signal=eqz_out_1)
    eqz_out_2 = np.convolve(desired_signal, coef_2.conjugate(), mode='full')[0:len(desired_signal)]
    distorted_signal_3 = non_lin_effect_sim.apply_channel(signal=eqz_out_2, pow_amp_mode=pow_amp_mode)

    coef_3 = tx_eqz_des.train_coef(unwanted_signal=distorted_signal_3, n_taps=n_taps, desired_signal=eqz_out_2)
    eqz_out_3 = np.convolve(desired_signal, coef_3.conjugate(), mode='full')[0:len(desired_signal)]
    distorted_signal_4 = non_lin_effect_sim.apply_channel(signal=eqz_out_3, pow_amp_mode=pow_amp_mode)

    coef_4 = tx_eqz_des.train_coef(unwanted_signal=distorted_signal_4, n_taps=n_taps, desired_signal=eqz_out_3)
    eqz_out_4 = np.convolve(desired_signal, coef_4.conjugate(), mode='full')[0:len(desired_signal)]
    distorted_signal_5 = non_lin_effect_sim.apply_channel(signal=eqz_out_4, pow_amp_mode=pow_amp_mode)

    print(non_lin_effect_sim.find_perf_metric(desired_signal))
    print(non_lin_effect_sim.find_perf_metric(distorted_signal_0))
    print(non_lin_effect_sim.find_perf_metric(distorted_signal_1))
    print(non_lin_effect_sim.find_perf_metric(distorted_signal_2))
    print(non_lin_effect_sim.find_perf_metric(distorted_signal_3))
    print(non_lin_effect_sim.find_perf_metric(distorted_signal_4))
    print(non_lin_effect_sim.find_perf_metric(distorted_signal_5))

    # mf_out = non_lin_effect_sim.tx_eqz_des_by_chirp.gen_mf_out(distorted_signal)
    # figure = make_subplots(rows=1, cols=1)
    # figure.add_trace(go.Scatter(y=mf_out), row=1, col=1)
    # figure.write_html("mf_out.html")

    # plot_filter_freq_resp(non_lin_effect_sim.b_list[0], non_lin_effect_sim.a_list[0])

    # input = non_lin_effect_sim.tx_eqz_des_by_chirp.desired_signal
    # output = lfilter(b, a, input)
    # figure = make_subplots(rows=2, cols=1)
    # figure.add_trace(go.Scatter(y=input.real), row=1, col=1)
    # figure.add_trace(go.Scatter(y=input.imag), row=2, col=1)
    # figure.add_trace(go.Scatter(y=output.real), row=1, col=1)
    # figure.add_trace(go.Scatter(y=output.imag), row=2, col=1)
    # figure.update_yaxes(range=[-1.5, 1.5], row=1, col=1)
    # figure.update_yaxes(range=[-1.5, 1.5], row=2, col=1)
    # figure.write_html("filter_io.html")

    # figure = make_subplots(rows=2, cols=1)
    # figure.add_trace(go.Scatter(y=distorted_signal.real, line=dict(width=2)), row=1, col=1)
    # figure.add_trace(go.Scatter(y=distorted_signal.imag, line=dict(width=2)), row=2, col=1)
    # figure.update_layout(
    #     xaxis=dict(title="sample"),       
    #     yaxis=dict(title="amplitude (real part)"),
    #     xaxis2=dict(title="sample"),      
    #     yaxis2=dict(title="amplitude (imaginary part)"),
    #     font=dict(size=20)
    # )
    # figure.write_html("simulation_result.html")

    
    print("DONE")


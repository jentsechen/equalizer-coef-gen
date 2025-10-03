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
import padasip as pa

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

    print(non_lin_effect_sim.find_perf_metric(desired_signal))

    # print("method 1")
    # distorted_signal = non_lin_effect_sim.apply_channel(signal=desired_signal, pow_amp_mode=pow_amp_mode)
    # print(non_lin_effect_sim.find_perf_metric(distorted_signal))
    # eqz_out = desired_signal
    # coef_hist_method_1 = []
    # for i in range(10):
    #     coef = tx_eqz_des.train_coef(unwanted_signal=distorted_signal, n_taps=n_taps, desired_signal=eqz_out)
    #     eqz_out = np.convolve(desired_signal, coef.conjugate(), mode='full')[0:len(desired_signal)]
    #     distorted_signal = non_lin_effect_sim.apply_channel(signal=eqz_out, pow_amp_mode=pow_amp_mode)
    #     print(non_lin_effect_sim.find_perf_metric(distorted_signal))
    #     coef_hist_method_1.append(coef)

    print("method 2")
    mu = 0.05
    distorted_signal = non_lin_effect_sim.apply_channel(signal=desired_signal, pow_amp_mode=pow_amp_mode)
    print(non_lin_effect_sim.find_perf_metric(distorted_signal))
    coef = tx_eqz_des.train_coef(unwanted_signal=distorted_signal, n_taps=n_taps)
    # coef = np.zeros(n_taps)+1j*np.zeros(n_taps)
    # coef[0] = 1
    # coef /= np.linalg.norm(coef)
    coef_hist, dis_sig_hist = [], []
    ref_norm = 0
    for iter in range(10):
        distorted_signal_pad = np.pad(distorted_signal, (n_taps, 0))
        coef_tmp = np.zeros(n_taps)+1j*np.zeros(n_taps)
        for i in range(4100):
            error = np.conj(desired_signal[i]) - np.vdot(distorted_signal_pad[i:(i+n_taps)], coef)
            coef_tmp += (error*distorted_signal_pad[i:(i+n_taps)])
        coef += (mu*coef_tmp/np.linalg.norm(coef_tmp))
        coef_hist.append(coef)
        eqz_out = np.convolve(desired_signal, coef, mode='full')[0:len(desired_signal)]
        distorted_signal = non_lin_effect_sim.apply_channel(signal=eqz_out, pow_amp_mode=pow_amp_mode)
        dis_sig_hist.append(distorted_signal)
        if iter==0:
            ref_norm = np.linalg.norm(distorted_signal-desired_signal)
        print(non_lin_effect_sim.find_perf_metric(distorted_signal), np.linalg.norm(distorted_signal-desired_signal)/ref_norm)

    # coef_hist, dis_sig_hist = [], []
    # for iter in range(10):
    #     coef_tmp = np.zeros(n_taps)+1j*np.zeros(n_taps)
    #     for i in range(4100):
    #         error = desired_signal[i] - distorted_signal[i]
    #         if i+n_taps>4100:
    #             # des = np.concatenate([desired_signal[i::], np.zeros(i+n_taps-4100)])
    #             des = np.pad(desired_signal[i::], (i+n_taps-4100, 0))
    #         else:
    #             des = desired_signal[i:(i+n_taps)]
    #         # coef_tmp += e_n * np.conj(x_n)
    #         # coef_tmp += (mu*error*np.conj(des))
    #         coef_tmp += (error*np.conj(des))
    #     # coef += coef_tmp/sum(coef_tmp)
    #     coef += (mu*coef_tmp/sum(coef_tmp))
    #     coef_hist.append(coef)
    #     eqz_out = np.convolve(desired_signal, coef, mode='full')[0:len(desired_signal)]
    #     distorted_signal = non_lin_effect_sim.apply_channel(signal=eqz_out, pow_amp_mode=pow_amp_mode)
    #     dis_sig_hist.append(distorted_signal)
    #     print(non_lin_effect_sim.find_perf_metric(distorted_signal))

    # print("distorted signal")
    # distorted_signal = non_lin_effect_sim.apply_channel(signal=desired_signal, pow_amp_mode=pow_amp_mode)
    # figure = make_subplots(rows=2, cols=1)
    # figure.add_trace(go.Scatter(y=desired_signal.real), row=1, col=1)
    # figure.add_trace(go.Scatter(y=desired_signal.imag), row=2, col=1)
    # figure.add_trace(go.Scatter(y=distorted_signal.real / 645), row=1, col=1)
    # figure.add_trace(go.Scatter(y=distorted_signal.imag / 645), row=2, col=1)
    # for i in range(int(len(dis_sig_hist)/3)):
    #     figure.add_trace(go.Scatter(y=dis_sig_hist[i*3].real / 645), row=1, col=1)
    #     figure.add_trace(go.Scatter(y=dis_sig_hist[i*3].imag / 645), row=2, col=1)
    # figure.write_html("dis_sig.html")

    data = {"d": {}, "v0": {}, "v1": {}, "v5": {}, "v9": {}}
    data["d"] = {"re": desired_signal.real.tolist(), "im": desired_signal.imag.tolist()}
    data["v0"] = {"re": (distorted_signal.real/645).tolist(), "im": (distorted_signal.imag/645).tolist()}
    data["v1"] = {"re": (dis_sig_hist[0].real/645).tolist(), "im": (dis_sig_hist[0].imag/645).tolist()}
    data["v5"] = {"re": (dis_sig_hist[4].real/645).tolist(), "im": (dis_sig_hist[4].imag/645).tolist()}
    data["v9"] = {"re": (dis_sig_hist[8].real/645).tolist(), "im": (dis_sig_hist[8].imag/645).tolist()}
    with open("sim_result.json", "w", encoding="UTF-8") as f:
        json.dump(data, f)

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


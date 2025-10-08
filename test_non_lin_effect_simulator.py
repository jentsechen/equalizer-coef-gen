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

def lms_test_one_iter():
    pslr_db_list, irw_m_list = [], []
    non_lin_effect_sim = NonLinEffectSimulator(n_rf_path=40, rf_path_p_to_p_diff_en=True, rand_cpx_gain_en=True)
    n_taps = 31
    pow_amp_mode = PowAmpMode.NonLinLut
    # pow_amp_mode = PowAmpMode.Linear
    tx_eqz_des = non_lin_effect_sim.tx_eqz_des_by_chirp
    desired_signal = tx_eqz_des.desired_signal
    # print(non_lin_effect_sim.find_perf_metric(desired_signal))
    mu = 0.01
    distorted_signal_0 = non_lin_effect_sim.apply_channel(signal=desired_signal, pow_amp_mode=pow_amp_mode)
    distorted_signal_0_pad = np.pad(distorted_signal_0, (n_taps, 0))
    # 
    pslr_db, irw_m = non_lin_effect_sim.find_perf_metric(distorted_signal_0)
    print(pslr_db, irw_m)
    pslr_db_list.append(pslr_db)
    irw_m_list.append(irw_m)
    # 
    coef = tx_eqz_des.train_coef(unwanted_signal=distorted_signal_0, n_taps=n_taps)
    eqz_out = np.convolve(desired_signal, coef, mode='full')[0:len(desired_signal)]
    distorted_signal = non_lin_effect_sim.apply_channel(signal=eqz_out, pow_amp_mode=pow_amp_mode)
    # 
    pslr_db, irw_m = non_lin_effect_sim.find_perf_metric(distorted_signal)
    print(pslr_db, irw_m)
    pslr_db_list.append(pslr_db)
    irw_m_list.append(irw_m)
    # 
    coef_hist = []
    # ref_norm = 0
    for iter in range(10):
        coef_tmp = np.zeros(n_taps)+1j*np.zeros(n_taps)
        for i in range(4100):
            coef_tmp += ((distorted_signal[i] - np.conj(desired_signal[i]))*distorted_signal_0_pad[i:(i+n_taps)])
        coef -= (mu*coef_tmp/np.linalg.norm(coef_tmp))
        coef_hist.append(coef)
        eqz_out = np.convolve(desired_signal, coef, mode='full')[0:len(desired_signal)]
        distorted_signal = non_lin_effect_sim.apply_channel(signal=eqz_out, pow_amp_mode=pow_amp_mode)
        # if iter==0:
        #     ref_norm = np.linalg.norm(distorted_signal-desired_signal)
        # print(non_lin_effect_sim.find_perf_metric(distorted_signal), np.linalg.norm(distorted_signal-desired_signal)/ref_norm)
        # print(non_lin_effect_sim.find_perf_metric(distorted_signal))
        # 
        pslr_db, irw_m = non_lin_effect_sim.find_perf_metric(distorted_signal)
        print(pslr_db, irw_m)
        pslr_db_list.append(pslr_db)
        irw_m_list.append(irw_m)
        #
    return pslr_db_list, irw_m_list

if __name__ == "__main__":
    pslr_db_set, irw_m_set = [], []
    for n_iter in range(10):    
        print("iteration: {}".format(n_iter))
        pslr_db_list, irw_m_list = lms_test_one_iter()
        pslr_db_set.append(pslr_db_list)
        irw_m_set.append(irw_m_list)

    with open("result.json", "w", encoding="UTF-8") as f:
        json.dump({"pslr_db": pslr_db_set, "irw_m": irw_m_set}, f)
    
    print("DONE")


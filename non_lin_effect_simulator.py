import numpy as np
from scipy.signal import remez
from tx_equalizer_design import TxEqzDesByChirp
from enum import Enum, auto
from pow_amp_non_lin_rapp import PowAmpNonLinRapp
from pow_amp_non_lin_lut import PowAmpNonLinLut
from pow_amp_non_lin_common import PowAmpNonLinCommon
from scipy.signal import lfilter, cheby1

class PowAmpMode(Enum):
    Linear, NonLinRapp, NonLinLut = auto(), auto(), auto()    

class NonLinEffectSimulator():
    def __init__(self, n_rf_path, rf_path_p_to_p_diff_en, rand_cpx_gain_en):
        cutoff = 650/1250/2
        self.n_rf_path = n_rf_path
        self.rf_path_imp_resp_list = []
        self.a_list, self.b_list = [], []
        for _ in range(self.n_rf_path):
            if rf_path_p_to_p_diff_en == True:
                ripple_factor = abs(np.random.rand()*10)
                filter_imp_resp = remez(numtaps=11, bands=[0, cutoff, cutoff+0.05+abs(np.random.rand())*0.05, 0.5], desired=[1, 0], weight=[1, ripple_factor], fs=1.0)
                b, a = cheby1(N=5, rp=1, Wn=(800e6+np.random.rand()*10e6)/1.25e9, btype='lowpass')
            else:
                filter_imp_resp = remez(numtaps=11, bands=[0, cutoff, cutoff+0.05, 0.5], desired=[1, 0], weight=[1, 10], fs=1.0) 
                b, a = cheby1(N=5, rp=1, Wn=800e6/1.25e9, btype='lowpass')
            self.rf_path_imp_resp_list.append(filter_imp_resp / sum(filter_imp_resp))
            self.a_list.append(a)
            self.b_list.append(b)
        self.tx_eqz_des_by_chirp = TxEqzDesByChirp()
        self.pow_amp_non_lin_rapp = PowAmpNonLinRapp()
        self.pow_amp_non_lin_lut = PowAmpNonLinLut()
        self.pow_amp_non_lin_common = PowAmpNonLinCommon()
        self.cpx_gain = self.gen_cpx_gain(rand_cpx_gain_en)

    def apply_one_path(self, signal, pow_amp_mode, rf_path_imp_resp, cpx_gain, b, a):
        # rf_model_out = np.convolve(signal, rf_path_imp_resp, mode='same') * cpx_gain
        rf_model_out = lfilter(b, a, signal) * cpx_gain
        if pow_amp_mode == PowAmpMode.Linear:
            return rf_model_out * self.pow_amp_non_lin_rapp.gain
        elif pow_amp_mode == PowAmpMode.NonLinRapp:
            pow_amp_out = []
            for r in rf_model_out:
                pow_amp_out.append(self.pow_amp_non_lin_rapp.apply(r))
            return np.array(pow_amp_out)
        elif pow_amp_mode == PowAmpMode.NonLinLut:
            pow_amp_out = []
            for r in rf_model_out:
                pow_amp_out.append(self.pow_amp_non_lin_lut.apply(r))
            return np.array(pow_amp_out)
        else:
            print("pow_amp_mode is not supported!")

    def apply_channel(self, signal, pow_amp_mode):
        output = self.apply_one_path(signal, pow_amp_mode, self.rf_path_imp_resp_list[0], self.cpx_gain[0], self.b_list[0], self.a_list[0])
        for i in range(1, self.n_rf_path):
            output += self.apply_one_path(signal, pow_amp_mode, self.rf_path_imp_resp_list[i], self.cpx_gain[i], self.b_list[i], self.a_list[i])
        return output
    
    def find_perf_metric(self, signal):
        mf_out = self.tx_eqz_des_by_chirp.gen_mf_out(signal)
        pslr_db = self.tx_eqz_des_by_chirp.find_pslr_db(mf_out)
        irw_m = self.tx_eqz_des_by_chirp.find_irw_m(mf_out)
        return pslr_db, irw_m

    def gen_cpx_gain(self, rand_cpx_gain_en):
        if rand_cpx_gain_en == True:
            gain_var_db = np.random.uniform(low=-1.5, high=1.5, size=self.n_rf_path)
            gain_lin = []
            for g in gain_var_db:
                gain_lin.append(self.pow_amp_non_lin_common.power_dbm_to_voltage_peak(6.88 + g))
            phase_deg = np.random.uniform(low=-10.0, high=10.0, size=self.n_rf_path)
            return np.array(gain_lin) * np.exp(1j * (phase_deg / 180 * np.pi))
        else:
            return np.ones(self.n_rf_path) * self.pow_amp_non_lin_common.power_dbm_to_voltage_peak(6.88)



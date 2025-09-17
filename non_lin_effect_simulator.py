import numpy as np
from scipy.signal import remez
from tx_equalizer_design import TxEqzDesByChirp
from enum import Enum, auto
from pow_amp_non_lin import PowAmpNonLinRapp

class PowAmpMode(Enum):
    Linear, NonLinRapp = auto(), auto()    

class NonLinEffectSimulator():
    def __init__(self, n_rf_path, rf_path_p_to_p_diff_en):
        cutoff = 650/1250/2
        self.n_rf_path = n_rf_path
        self.rf_path_imp_resp_list = []
        for i in range(self.n_rf_path):
            if rf_path_p_to_p_diff_en == True:
                ripple_factor = abs(np.random.rand()*10)
                filter_imp_resp = remez(numtaps=11, bands=[0, cutoff, cutoff+0.05+abs(np.random.rand())*0.05, 0.5], desired=[1, 0], weight=[1, ripple_factor], fs=1.0)
            else:
                filter_imp_resp = remez(numtaps=11, bands=[0, cutoff, cutoff+0.05, 0.5], desired=[1, 0], weight=[1, 10], fs=1.0) 
            self.rf_path_imp_resp_list.append(filter_imp_resp / sum(filter_imp_resp))
        self.tx_eqz_des_by_chirp = TxEqzDesByChirp()
        self.pow_amp_non_lin_rapp = PowAmpNonLinRapp()
        self.pa_in_volt_peak = self.pow_amp_non_lin_rapp.power_dbm_to_voltage_peak(6.88)

    def apply_one_path(self, signal, pow_amp_mode, rf_path_imp_resp):
        rf_model_out = np.convolve(signal, rf_path_imp_resp, mode='same')*self.pa_in_volt_peak
        if pow_amp_mode == PowAmpMode.Linear:
            return rf_model_out * self.pow_amp_non_lin_rapp.gain
        elif pow_amp_mode == PowAmpMode.NonLinRapp:
            pow_amp_out = []
            for r in rf_model_out:
                pow_amp_out.append(self.pow_amp_non_lin_rapp.apply(r))
            return np.array(pow_amp_out)            
        else:
            print("pow_amp_mode is not supported!")

    def apply_channel(self, signal, pow_amp_mode):
        output = self.apply_one_path(signal, pow_amp_mode, self.rf_path_imp_resp_list[0])
        for i in range(1, self.n_rf_path):
            output += self.apply_one_path(signal, pow_amp_mode, self.rf_path_imp_resp_list[i])
        return output
    
    def find_perf_metric(self, signal):
        mf_out = self.tx_eqz_des_by_chirp.gen_mf_out(signal)
        pslr_db = self.tx_eqz_des_by_chirp.find_pslr_db(mf_out)
        irw_m = self.tx_eqz_des_by_chirp.find_irw_m(mf_out)
        return pslr_db, irw_m

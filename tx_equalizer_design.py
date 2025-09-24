import numpy as np
import json
from scipy.signal import resample_poly
from chisel3cr.common import qt, eSign, eMSB, eLSB

class TxEqualizerDesign():
    def __init__(self, desired_signal, qntz_format: qt):
        self.desired_signal = desired_signal
        self.qntz_format = qntz_format
        self.font_size = 20
        self.matched_filter_up_factor = 200

    def __resample(self, data):
        return resample_poly(data, up=5, down=4)
    
    def __matched_filter(self, rx_waveform):
        return abs(np.convolve(rx_waveform, self.desired_signal.conjugate(), mode='valid'))
    
    def __find_start_idx(self, rx_waveform):
        return np.argmax(self.__matched_filter(rx_waveform))
    
    def proc_meas_to_train(self, meas_data):
        resmp_data = self.__resample(meas_data)
        start_idx = self.__find_start_idx(resmp_data)
        return resmp_data[start_idx:(start_idx+len(self.desired_signal))]

    def train_coef(self, unwanted_signal, n_taps, desired_signal=None, eps=0.0):
        if desired_signal is None:
            desired_signal = self.desired_signal
        n_group = min(len(unwanted_signal), len(desired_signal)) - (n_taps - 1)

        R = np.zeros((n_taps, n_taps), dtype=complex)
        p = np.zeros(n_taps, dtype=complex)
        for g in range(n_group):
            idx_sta = g
            idx_end = idx_sta + n_taps
            u = unwanted_signal[idx_sta:idx_end][::-1]        
            d = desired_signal[idx_sta + (n_taps - 1)]      

            R += np.outer(u, np.conj(u))             
            p += u * np.conj(d)                      

        if eps > 0:
            R = R + eps * np.eye(n_taps)

        w = np.linalg.solve(R, p)
        return w / sum(w)
    
    def gen_eqz_out(self, meas_data, n_taps, sim_en=False):
        if sim_en == True:
            train_data = meas_data
        else:
            train_data = self.proc_meas_to_train(meas_data)
        coef = self.train_coef(train_data, n_taps)
        eqz_out = np.convolve(self.desired_signal, coef.conjugate(), mode='full')[0:len(self.desired_signal)]
        # eqz_dly = int((n_taps-1)/2)
        # eqz_out = np.convolve(train_data, coef.conjugate())[eqz_dly:len(self.desired_signal)+eqz_dly]
        if sim_en == True:
            return eqz_out
        else:
            eqz_out_fix = []
            for i in range(len(eqz_out)):
                re = self.qntz_format.apply(eqz_out[i].real)
                im = self.qntz_format.apply(eqz_out[i].imag)
                eqz_out_fix.append(re + 1j * im)
            return np.array(eqz_out_fix)
    
    def gen_mf_out(self, data):
        mf_out = np.convolve(data, np.conj(self.desired_signal))
        center, width = np.argmax(mf_out), 20
        # center, width = len(self.desired_signal), 20
        assert width%2 == 0
        mf_out_up_smp = resample_poly(mf_out[(center-int(width/2)):(center+int(width/2))], up=self.matched_filter_up_factor, down=1)
        return 20*np.log10(abs(mf_out_up_smp))

    def gen_eqz_mf_out(self, meas_data, n_taps):
        return self.gen_mf_out(self.gen_eqz_out(meas_data, n_taps))
    
    def gen_meas_mf_out(self, meas_data):
        return self.gen_mf_out(self.proc_meas_to_train(meas_data))
    
    def find_pslr_db(self, mf_out):
        main_lobe_peak_index = int(np.argmax(mf_out))
        side_lobe_peak_mag_db = []
        for i in range(1, len(mf_out-1)):
            if (i != main_lobe_peak_index) and (mf_out[i] > mf_out[i-1]) and (mf_out[i] >= mf_out[i+1]):
                side_lobe_peak_mag_db.append(mf_out[i])
        return mf_out[main_lobe_peak_index] - max(side_lobe_peak_mag_db)

    def find_irw_m(self, mf_out):
        mf_out_norm = mf_out - np.max(mf_out)
        peak_index = np.argmax(mf_out_norm)
        left_index = peak_index
        while left_index > 0 and mf_out_norm[left_index] >= -3:
            left_index -= 1

        right_index = peak_index
        while right_index < len(mf_out_norm) - 1 and mf_out_norm[right_index] >= -3:
            right_index += 1

        irw_samples = right_index - left_index
        return irw_samples / self.matched_filter_up_factor * 3e8 / 1.25e9 / 2
    
class TxEqzDesByChirp(TxEqualizerDesign):
    def __init__(self):
        with open("./training_sig/sig_chirp_s0_15.json", "r", encoding="UTF-8") as f:
            desired_signal_j = json.load(f)
        desired_signal = desired_signal_j["re"] + 1j * np.array(desired_signal_j["im"])
        qntz_format = qt(sign=eSign.Signed, int_bit=0, frac_bit=15, msb=eMSB.Sat, lsb=eLSB.Rnd)
        super().__init__(desired_signal, qntz_format)

class TxEqzDesByOfdm(TxEqualizerDesign):
    def __init__(self):
        with open("./training_sig/sig_ofdm_s1_14.json", "r", encoding="UTF-8") as f:
            desired_signal_j = json.load(f)
        desired_signal = (desired_signal_j["re"] + 1j * np.array(desired_signal_j["im"]))[0:4096]
        qntz_format = qt(sign=eSign.Signed, int_bit=1, frac_bit=14, msb=eMSB.Sat, lsb=eLSB.Rnd)
        super().__init__(desired_signal, qntz_format)

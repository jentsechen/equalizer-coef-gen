import numpy as np
from .utility_lib import PAR, move_files
from .sp import single_tone, mixer, expj, add_noise_dbm, mixer, expj, power_est_dB
from .plot import plot_figure


class Simple_AFE_Model():
    def __init__(self, **args):
        self.set_par(args)
        self.noise_dbm_mhz = -173.8 + self.par.lna.nf + 60
        self.spur_L = 0
        self.spur_F = 0
        self.noise_dbm = self.noise_dbm_mhz + 10 * np.log10(
            self.par.sys.fs_mhz)

    def set_par(self, args):
        par = PAR(args)
        par.set('noise_en', True)
        par.set('lna.gain', 54.0)
        par.set('lna.nf', 3.5)
        par.set('sys.fs_mhz', 160)
        par.set('lpf.gain', 12)
        par.set('cfo_khz', 0)
        par.set('spur.power_dbm', None)
        par.set('spur.freq_mhz', None)
        par.set('adc.gain.dbm2dbv', -13, force=True)
        par.set('adc.gain.vpp2x', 12, force=True)  # +/-0.5->+/-4
        par.set('adc.gain.all',
                par.get('adc.gain.dbm2dbv') + par.get('adc.gain.vpp2x'),
                force=True)
        par.set('time.delay_us', 0)
        par.set('time.tail_us', 0)
        par.save('Simple_AFE_Model.json')
        self.par = par.get_class()

    def mixer(self, a, cfo_khz):
        return mixer(a, self.par.sys.fs_mhz * 1e6, 1e3 * cfo_khz)

    def delay_tail(self, a, delay_us, tail_us):
        delay_sample = (int)(delay_us * self.par.sys.fs_mhz)
        tail_sample = (int)(tail_us * self.par.sys.fs_mhz)

        if delay_sample or tail_sample:
            out = np.zeros(shape=(a.shape[0],
                                  a.shape[1] + delay_sample + tail_sample),
                           dtype=np.complex128)
            out[:, delay_sample:delay_sample + a.shape[1]] = a
            return out
        else:
            return a

    def add_noise(self, a, noise_en):
        return add_noise_dbm(a, self.noise_dbm) if noise_en else a

    def gen_spur(self, L, F):
        if self.spur_L != L or self.spur_F != F:
            self.spur = single_tone(F, self.par.sys.fs_mhz, L)
            self.spur_L = L
            self.spur_F = F
        return self.spur

    def add_spur(self, a, power=None, freq_mhz=None):
        if power is None:
            power = self.par.spur.power_dbm
        if freq_mhz is None:
            freq_mhz = self.par.spur.freq_mhz
        if (power is None) or (freq_mhz is None):
            return a
        else:
            spur_gain = 10**(power / 20.0)
            spur = self.gen_spur(a.shape[1], freq_mhz)
            out = np.zeros(a.shape, dtype=np.complex128)
            for i in range(a.shape[0]):
                out[i] = a[i] + np.multiply(
                    spur_gain * expj(np.random.uniform()), spur)
            return out

    def amplify(self, a, lna_gain, lpf_gain):
        total_gain_dB = lna_gain + lpf_gain + self.par.adc.gain.all
        total_gain = 10**(total_gain_dB / 20.0)
        out = np.multiply(total_gain, a)
        return out

    def run(self,
            rx_in,
            cfo_khz=None,
            delay_us=None,
            tail_us=None,
            spur_power=None,
            spur_freq_mhz=None,
            noise_en=None,
            lna_gain=None,
            lpf_gain=None,
            **vargs):
        par = self.par
        mixer_out = self.mixer(rx_in, cfo_khz or par.cfo_khz)
        delay_out = self.delay_tail(mixer_out, delay_us or par.time.delay_us,
                                    tail_us or par.time.tail_us)
        spur_out = self.add_spur(delay_out, spur_power, spur_freq_mhz)
        noise_out = self.add_noise(spur_out, noise_en or par.noise_en)
        amplify_out = self.amplify(noise_out, lna_gain or par.lna.gain,
                                   lpf_gain or par.lpf.gain)
        return amplify_out

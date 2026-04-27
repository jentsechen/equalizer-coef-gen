import numpy as np
import json
import os
import matplotlib.pyplot as plt
# from chisel3cr.common import qt, eSign, eMSB, eLSB
from scipy import signal
from dataclasses import dataclass


@dataclass
class FilterDesign:
    fs_mhz: float = 750.0
    f_pass_mhz: float = 100.0
    f_stop_mhz: float = 0.0
    Apass_dB: float = 0.05
    Astop_dB: float = 50
    n_taps: int = 65
    response: np.ndarray = None

    def __post_init__(self):

        def generate_response():
            err_pb = (
                1 - 10**(-self.Apass_dB / 20)
            ) / 2  # /2 is not part of the article above, but makes the result consistent with pyFDA
            err_sb = 10**(-self.Astop_dB / 20)

            w_pb = 1 / err_pb
            w_sb = 1 / err_sb

            # Calculate that FIR coefficients
            # print(self.n_taps, self.f_pass_mhz, self.f_stop_mhz)
            self.response = signal.remez(
                self.n_taps,
                [
                    0., self.f_pass_mhz / self.fs_mhz,
                    min([self.f_stop_mhz / self.fs_mhz, 0.475]), 0.5
                ],  # Filter inflection points
                [
                    1, 0
                ],  # Desired gain for each of the bands: 1 in the pass band, 0 in the stop band
                [w_pb, w_sb
                 ]  # weights used to get the right ripple and attenuation
            )

        if self.f_stop_mhz < self.f_pass_mhz:
            self.f_stop_mhz = self.fs_mhz / 2

        generate_response()

@dataclass
class FirlsFilterDesign:
    fs_mhz: float = 750.0
    n_taps: int = 73
    freq_mhz: np.ndarray = None   # frequency points of the desired response
    gain: np.ndarray = None        # desired gain at each frequency point
    f_stop_mhz: float = 0.0       # stopband start in MHz; 0 = no stopband constraint
    w_stop: float = 100.0          # stopband weight relative to passband
    response: np.ndarray = None

    def __post_init__(self):
        freq = np.asarray(self.freq_mhz, dtype=float)
        gain = np.asarray(self.gain, dtype=float)

        # Represent the dense desired response as piecewise-linear bands:
        # each adjacent pair (freq[k], freq[k+1]) becomes one band with gain
        # linearly interpolated between gain[k] and gain[k+1].
        # scipy firls allows touching band edges (zero-width transition bands).
        n = len(freq)
        bands = np.empty(2 * (n - 1))
        bands[0::2] = freq[:-1]
        bands[1::2] = freq[1:]
        desired = np.empty(2 * (n - 1))
        desired[0::2] = gain[:-1]
        desired[1::2] = gain[1:]
        weights = np.ones(n - 1)

        nyq = self.fs_mhz / 2.0
        if 0 < self.f_stop_mhz < nyq:
            bands = np.append(bands, [self.f_stop_mhz, nyq])
            desired = np.append(desired, [0.0, 0.0])
            weights = np.append(weights, self.w_stop)

        self.response = signal.firls(
            self.n_taps, bands, desired, weight=weights, fs=self.fs_mhz
        )


# class eSarRxFastConvMode(Enum):
#     SingleBeam, MultiBeam = auto(), auto()

def aaf_coef_gen(f_pass_mhz=325,
                 bypass=False,
                 mode=False):
    fs_mhz = 750
    ftransient_min = 25
    if mode:
        f_pass_mhz /= 4
        fs_mhz /= 4
        ftransient_min /= 4
    f_delta = np.max([f_pass_mhz * 0.2, ftransient_min])
    f_stop_mhz = np.min([f_pass_mhz + f_delta, (float)(fs_mhz / 2)])
    n_tapsAAF = 73
    if bypass:
        zero_seq = np.zeros(int((n_tapsAAF - 1) / 2))
        aaf_coef = np.concatenate((zero_seq, np.ones(1), zero_seq))
    else:
        aaf_coef = FilterDesign(
            fs_mhz=fs_mhz,
            f_pass_mhz=f_pass_mhz,
            f_stop_mhz=f_stop_mhz,
            Apass_dB=0.02,
            Astop_dB=60,
            n_taps=n_tapsAAF).response
    return aaf_coef


def aaf_coef_save_json():
    aaf_coef = []
    aaf_coef.append(aaf_coef_gen(325).tolist())
    aaf_coef.append(aaf_coef_gen(162.5).tolist())
    aaf_coef.append(aaf_coef_gen(140).tolist())
    aaf_coef.append(aaf_coef_gen(110).tolist())
    aaf_coef.append(aaf_coef_gen(75).tolist())
    aaf_coef.append(aaf_coef_gen(55).tolist())
    aaf_coef.append(aaf_coef_gen(40).tolist())
    aaf_coef.append(aaf_coef_gen(25).tolist())
    aaf_coef.append(aaf_coef_gen(12.5).tolist())
    para = {'aaf_coef': []}
    para['aaf_coef'] = aaf_coef
    out_dir = os.path.join(os.path.dirname(__file__), "aaf_freq_resp")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "aaf_imp_resp.json"),
              "w",
              encoding='UTF-8') as f:
        json.dump(para, f)


def plot_aaf_freq_resp_overlay(aaf_sel=0):
    import plotly.graph_objects as go

    freq_resp_path = os.path.join(os.path.dirname(__file__), "aaf_freq_resp", "aaf_freq_resp_256.json")
    coef_path = os.path.join(os.path.dirname(__file__), "aaf_freq_resp", "aaf_imp_resp.json")

    with open(freq_resp_path) as f:
        freq_resp_data = json.load(f)
    with open(coef_path) as f:
        coef_data = json.load(f)

    re = np.array(freq_resp_data['re'][aaf_sel])
    im = np.array(freq_resp_data['im'][aaf_sel])
    freq_resp_db = 20 * np.log10(np.abs(re + 1j * im) + 1e-12)

    coef = np.array(coef_data['aaf_coef'][aaf_sel])
    fft_db = 20 * np.log10(np.abs(np.fft.fft(coef, 256)) + 1e-12)

    fs_mhz = 750.0
    freq_mhz = np.arange(256) * fs_mhz / 256

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq_mhz.tolist(), y=freq_resp_db.tolist(), name='aaf_freq_resp_256[0]'))
    fig.add_trace(go.Scatter(x=freq_mhz.tolist(), y=fft_db.tolist(), name='256-FFT of aaf_coef[0]', line=dict(dash='dash')))
    fig.update_layout(xaxis_title="Frequency (MHz)", yaxis_title="Magnitude (dB)")

    out_dir = os.path.join(os.path.dirname(__file__), "figure")
    os.makedirs(out_dir, exist_ok=True)
    fig.write_html(os.path.join(out_dir, "aaf_resp", "aaf_freq_resp_overlay.html"))


if __name__ == "__main__":
    aaf_coef = aaf_coef_gen()
    aaf_coef_save_json()
    plot_aaf_freq_resp_overlay(aaf_sel=0)

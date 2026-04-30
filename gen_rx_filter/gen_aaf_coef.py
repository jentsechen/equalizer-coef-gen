import numpy as np
import json
import os
import matplotlib.pyplot as plt
# from chisel3cr.common import qt, eSign, eMSB, eLSB
from scipy import signal
from dataclasses import dataclass
from gen_rx_filter import gen_eqz_freq_resp, DistortedSig
import plotly.graph_objects as go

@dataclass
class FilterDesign:
    """Equiripple lowpass FIR filter designed via the Remez exchange algorithm.

    Attributes:
        fs_mhz: Sample rate in MHz.
        f_pass_mhz: Passband edge in MHz.
        f_stop_mhz: Stopband edge in MHz; defaults to Nyquist when less than f_pass_mhz.
        Apass_dB: Maximum passband ripple in dB.
        Astop_dB: Minimum stopband attenuation in dB.
        n_taps: Filter length (odd recommended).
        response: Computed FIR coefficients, populated by __post_init__.
    """

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
    """Least-squares FIR filter matching an arbitrary frequency-domain target.

    The desired response is specified as magnitude values at arbitrary frequency
    points and is linearly interpolated between them within each sub-band.
    An optional stopband band is appended from f_stop_mhz to Nyquist with
    zero desired gain and weight w_stop.

    Attributes:
        fs_mhz: Sample rate in MHz.
        n_taps: Filter length (odd required by firls).
        freq_mhz: Frequency points of the desired response in MHz.
        gain: Desired complex or real magnitude at each frequency point.
        f_stop_mhz: Stopband start in MHz; 0 disables the stopband constraint.
        w_stop: Stopband weight relative to passband bands (default 100).
        response: Computed FIR coefficients, populated by __post_init__.
    """

    fs_mhz: float = 750.0
    n_taps: int = 63
    freq_mhz: np.ndarray = None
    gain: np.ndarray = None
    f_stop_mhz: float = 0.0
    w_stop: float = 100.0
    response: np.ndarray = None

    def __post_init__(self):
        freq = np.asarray(self.freq_mhz, dtype=float)
        gain = np.abs(np.asarray(self.gain))

        if 0 < self.f_stop_mhz:
            mask = freq < self.f_stop_mhz
            freq, gain = freq[mask], gain[mask]

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


def aaf_coef_gen(f_pass_mhz=325,
                 bypass=False,
                 mode=False):
    """Generate AAF FIR coefficients for a given passband edge.

    Args:
        f_pass_mhz: Passband edge in MHz (at fs=750 MHz).
        bypass: If True, return an identity (delta) impulse instead of a filter.
        mode: If True, scale all frequencies by 1/4 (quarter-rate mode).

    Returns:
        ndarray of length 55 containing the FIR tap coefficients.
    """
    fs_mhz = 750
    ftransient_min = 25
    if mode:
        f_pass_mhz /= 4
        fs_mhz /= 4
        ftransient_min /= 4
    f_delta = np.max([f_pass_mhz * 0.2, ftransient_min])
    f_stop_mhz = np.min([f_pass_mhz + f_delta, (float)(fs_mhz / 2)])
    n_tapsAAF = 55
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
    """Generate AAF coefficients for all standard passband configurations and save to JSON.

    Writes aaf_imp_resp.json under gen_rx_filter/aaf_freq_resp/.
    """
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


def mag_db(H):
    """Return magnitude of H in dB."""
    return 20 * np.log10(np.abs(H) + 1e-12)


def phase_deg(H):
    """Return unwrapped phase of H in degrees."""
    return np.degrees(np.unwrap(np.angle(H)))


def compute_responses(f_pass_mhz, fs_mhz=750.0, n=256):
    """Compute frequency responses for the least-square EQZ filter, raw EQZ, and AAF×EQZ.

    Args:
        fs_mhz: Sample rate in MHz.
        n: FFT size used to evaluate the frequency responses.

    Returns:
        Tuple (H_fir, H_eqz, H_combined, freq_axis) where each H is a
        complex ndarray of length n in fftshift order and freq_axis is the
        corresponding frequency axis in MHz.
    """
    eqz_resp = gen_eqz_freq_resp(sig=DistortedSig.UCDC, n_taps=4)
    freq_mhz = np.arange(n // 2) * fs_mhz / n
    gain = eqz_resp[:n // 2]
    fir = FirlsFilterDesign(
        fs_mhz=fs_mhz,
        n_taps=63,
        freq_mhz=freq_mhz,
        gain=gain,
        f_stop_mhz=f_pass_mhz+25,
        w_stop=100,
    )
    H_fir = np.fft.fftshift(np.fft.fft(fir.response, n))
    H_eqz = np.fft.fftshift(eqz_resp)
    aaf_coef = aaf_coef_gen(f_pass_mhz=f_pass_mhz)
    H_combined = np.fft.fftshift(np.fft.fft(aaf_coef, n) * eqz_resp)
    freq_axis = np.fft.fftshift(np.arange(n) * fs_mhz / n - fs_mhz * (np.arange(n) >= n // 2))
    return H_fir, H_eqz, H_combined, freq_axis


def plot_magnitude_resp(freq_axis, H_fir, H_eqz, H_combined, out_dir, font_size):
    """Plot magnitude responses of the three filters and save to magnitude_resp.html."""
    fig = go.Figure()
    for H, name in [(H_eqz, 'equalizer'), (H_combined, 'comb. of AAF and equalizer'), (H_fir, 'least square')]:
        fig.add_trace(go.Scatter(x=freq_axis.tolist(), y=mag_db(H).tolist(), name=name))
    fig.update_layout(xaxis_title='frequency (MHz)', yaxis_title='magnitude (dB)', font=dict(size=font_size))
    fig.write_html(os.path.join(out_dir, 'magnitude_resp.html'))


def plot_phase_resp(freq_axis, H_fir, H_eqz, H_combined, out_dir, font_size):
    """Plot phase responses of the three filters and save to phase_resp.html."""
    fig = go.Figure()
    for H, name in [(H_eqz, 'equalizer'), (H_combined, 'comb. of AAF and equalizer'), (H_fir, 'least square')]:
        fig.add_trace(go.Scatter(x=freq_axis.tolist(), y=phase_deg(H).tolist(), name=name))
    fig.update_layout(xaxis_title='frequency (MHz)', yaxis_title='phase (deg)', font=dict(size=font_size))
    fig.write_html(os.path.join(out_dir, 'phase_resp.html'))


def plot_inband_error(freq_axis, H_fir, H_eqz, H_combined, out_dir, font_size, f_pass_mhz=325):
    """Plot in-band magnitude error of H_fir and H_combined relative to H_eqz.

    Args:
        f_pass_mhz: Passband edge in MHz; only frequencies within ±f_pass_mhz are shown.
    """
    pb = np.abs(freq_axis) < f_pass_mhz
    freq_pb = freq_axis[pb]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq_pb.tolist(), y=(mag_db(H_combined)[pb] - mag_db(H_eqz)[pb]).tolist(), name='comb. of AAF and equalizer'))
    fig.add_trace(go.Scatter(x=freq_pb.tolist(), y=(mag_db(H_fir)[pb] - mag_db(H_eqz)[pb]).tolist(), name='least square'))
    fig.update_layout(xaxis_title='frequency (MHz)', yaxis_title='magnitude error (dB)', font=dict(size=font_size))
    fig.write_html(os.path.join(out_dir, 'inband_error.html'))


def print_group_delay(freq_axis, H_fir, H_eqz, H_combined, fs_mhz, f_pass_mhz=325):
    """Print group delay of the three filters estimated from a linear phase fit over the passband.

    Args:
        f_pass_mhz: Passband edge in MHz used to select the fitting region.
    """
    pb = np.abs(freq_axis) < f_pass_mhz
    for label, H in [('equalizer', H_eqz), ('comb. of AAF and equalizer', H_combined), ('least square', H_fir)]:
        slope = np.polyfit(freq_axis[pb], phase_deg(H)[pb], 1)[0]  # deg/MHz
        gd_ns = -slope / 0.36
        gd_samples = gd_ns * fs_mhz * 1e-3
        print(f'{label}: group delay = {gd_ns:.1f} ns  ({gd_samples:.1f} samples)')


if __name__ == "__main__":
    fs_mhz = 750.0
    n = 256
    f_pass_mhz = 162.5
    H_fir, H_eqz, H_combined, freq_axis = compute_responses(f_pass_mhz, fs_mhz, n)

    out_dir = os.path.join(os.path.dirname(__file__), 'figure', 'aaf_resp')
    os.makedirs(out_dir, exist_ok=True)

    font_size = 25
    plot_magnitude_resp(freq_axis, H_fir, H_eqz, H_combined, out_dir, font_size)
    plot_phase_resp(freq_axis, H_fir, H_eqz, H_combined, out_dir, font_size)
    plot_inband_error(freq_axis, H_fir, H_eqz, H_combined, out_dir, font_size, f_pass_mhz)
    print_group_delay(freq_axis, H_fir, H_eqz, H_combined, fs_mhz)
    
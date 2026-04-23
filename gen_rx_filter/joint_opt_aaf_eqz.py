import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from tx_equalizer_design import TxEqzDesByChirp

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_signals(desired_signal, unwanted_signal):
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True)
    for sig, name in [(desired_signal, "desired"), (unwanted_signal, "unwanted")]:
        figure.add_trace(go.Scatter(y=sig.real, name=f"{name} real"), row=1, col=1)
        figure.add_trace(
            go.Scatter(y=sig.imag, name=f"{name} imag", showlegend=True), row=2, col=1
        )
        figure.add_trace(
            go.Scatter(y=np.abs(sig), name=f"{name} envelope", showlegend=True),
            row=3,
            col=1,
        )
    figure.update_layout(
        yaxis=dict(title="real"),
        yaxis2=dict(title="imag"),
        yaxis3=dict(title="envelope"),
        xaxis3=dict(title="sample"),
        font=dict(size=20),
        margin=dict(l=80, r=20, t=20, b=60),
    )
    fig_dir = os.path.join(_SCRIPT_DIR, "figure")
    os.makedirs(fig_dir, exist_ok=True)
    figure.write_html(os.path.join(fig_dir, "joint_opt_aaf_eqz.html"))

def gen_signals():
    tx_eqz = TxEqzDesByChirp(fs_is_750mhz=True)
    desired_signal = tx_eqz.desired_signal
    unwanted_signal = tx_eqz.proc_meas_to_train(np.load(os.path.join(_SCRIPT_DIR, "update_aaf_freq_resp/dc_distorted_sig.npy")), resample_meas=False)
    np.save(os.path.join(_SCRIPT_DIR, "update_aaf_freq_resp/desired_signal.npy"), desired_signal)
    np.save(os.path.join(_SCRIPT_DIR, "update_aaf_freq_resp/unwanted_signal.npy"), unwanted_signal)
    plot_signals(desired_signal, unwanted_signal)


def _bw_mask(n_fft: int, bw_ratio: float) -> np.ndarray:
    n_pass = round(n_fft * bw_ratio / 2)
    mask = np.zeros(n_fft)
    mask[:n_pass] = 1.0
    mask[n_fft - n_pass:] = 1.0
    return mask


def gen_eqz_freq_resp(n_taps: int = 256, eps: float = 0.0) -> np.ndarray:
    desired = np.load(os.path.join(_SCRIPT_DIR, "update_aaf_freq_resp/desired_signal.npy"))
    unwanted = np.load(os.path.join(_SCRIPT_DIR, "update_aaf_freq_resp/unwanted_signal.npy"))

    n_fft = 256
    bw_ratio = 650 / 750
    n_group = min(len(desired), len(unwanted)) - (n_taps - 1)

    U = unwanted[np.arange(n_group)[:, None] + np.arange(n_taps - 1, -1, -1)[None, :]]
    d = desired[n_taps - 1 : n_taps - 1 + n_group]

    R = U.conj().T @ U
    if eps > 0:
        R += eps * (np.trace(R).real / n_taps) * np.eye(n_taps)
    p = U.conj().T @ d
    h = np.linalg.solve(R, p)

    H = np.fft.fft(h, n=n_fft)
    H *= _bw_mask(n_fft, bw_ratio)
    return H


def plot_eqz_freq_resp(H: np.ndarray):
    n_fft = len(H)
    freq_axis = np.linspace(-375, 375, n_fft)
    H_shifted = np.fft.fftshift(H)
    mag_db = 20 * np.log10(np.maximum(np.abs(H_shifted), 1e-12))
    phase_deg = np.degrees(np.unwrap(np.angle(H_shifted)))

    figure = make_subplots(rows=2, cols=1, shared_xaxes=True)
    figure.add_trace(go.Scatter(x=freq_axis, y=mag_db, name="magnitude"), row=1, col=1)
    figure.add_trace(go.Scatter(x=freq_axis, y=phase_deg, name="phase"), row=2, col=1)
    figure.update_layout(
        yaxis=dict(title="magnitude (dB)"),
        yaxis2=dict(title="phase (deg)"),
        xaxis2=dict(title="frequency (MHz)"),
        font=dict(size=20),
        margin=dict(l=80, r=20, t=20, b=60),
    )
    fig_dir = os.path.join(_SCRIPT_DIR, "figure")
    os.makedirs(fig_dir, exist_ok=True)
    figure.write_html(os.path.join(fig_dir, "joint_opt_eqz_freq_resp.html"))


def gen_eqz_freq_resp_lstsq(n_taps: int = 256) -> np.ndarray:
    desired = np.load(os.path.join(_SCRIPT_DIR, "update_aaf_freq_resp/desired_signal.npy"))
    unwanted = np.load(os.path.join(_SCRIPT_DIR, "update_aaf_freq_resp/unwanted_signal.npy"))

    n_fft = 256
    bw_ratio = 650 / 750
    n_group = min(len(desired), len(unwanted)) - (n_taps - 1)

    U = unwanted[np.arange(n_group)[:, None] + np.arange(n_taps - 1, -1, -1)[None, :]]
    d = desired[n_taps - 1 : n_taps - 1 + n_group]

    h, _, _, _ = np.linalg.lstsq(U, d, rcond=None)

    H = np.fft.fft(h, n=n_fft)
    H *= _bw_mask(n_fft, bw_ratio)
    return H


def plot_sig_freq_resp_compare(H_ref: np.ndarray = None):
    desired = np.load(os.path.join(_SCRIPT_DIR, "update_aaf_freq_resp/desired_signal.npy"))
    unwanted = np.load(os.path.join(_SCRIPT_DIR, "update_aaf_freq_resp/unwanted_signal.npy"))

    n = len(desired)
    fs_mhz = 750
    freq_axis = np.fft.fftshift(np.fft.fftfreq(n, d=1 / fs_mhz))

    def mag_db(sig):
        m = np.fft.fftshift(20 * np.log10(np.maximum(np.abs(np.fft.fft(sig)), 1e-12)))
        return m - m[len(m) // 2]

    def compensated_sig(H):
        return np.convolve(unwanted, np.fft.ifft(H))[: len(unwanted)]

    figure = make_subplots(rows=1, cols=1)
    figure.add_trace(go.Scatter(x=freq_axis, y=mag_db(desired), name="desired"))
    figure.add_trace(go.Scatter(x=freq_axis, y=mag_db(unwanted), name="unwanted"))
    eps = 0.0
    for n_taps in [9, 32, 64]:
        H = gen_eqz_freq_resp(n_taps, eps=eps)
        figure.add_trace(go.Scatter(x=freq_axis, y=mag_db(compensated_sig(H)), name=f"Wiener {n_taps} taps ε={eps}"))

    if H_ref is not None:
        figure.add_trace(go.Scatter(x=freq_axis, y=mag_db(compensated_sig(H_ref)), name="compensated (ref)"))
    figure.update_layout(
        xaxis=dict(title="frequency (MHz)"),
        yaxis=dict(title="magnitude (dB)"),
        font=dict(size=20),
        margin=dict(l=80, r=20, t=20, b=60),
    )
    fig_dir = os.path.join(_SCRIPT_DIR, "figure")
    os.makedirs(fig_dir, exist_ok=True)
    figure.write_html(os.path.join(fig_dir, "sig_freq_resp_compare.html"))


def plot_sig_time_domain_compare(H_ref: np.ndarray = None):
    desired = np.load(os.path.join(_SCRIPT_DIR, "update_aaf_freq_resp/desired_signal.npy"))
    unwanted = np.load(os.path.join(_SCRIPT_DIR, "update_aaf_freq_resp/unwanted_signal.npy"))

    def compensated_sig(H):
        return np.convolve(unwanted, np.fft.ifft(H))[: len(unwanted)]

    norm = np.max(np.abs(desired))

    figure = make_subplots(rows=3, cols=1, shared_xaxes=True)
    for sig, name in [(desired, "desired"), (unwanted, "unwanted")]:
        sig = sig / norm
        figure.add_trace(go.Scatter(y=sig.real, name=f"{name}"), row=1, col=1)
        figure.add_trace(go.Scatter(y=sig.imag, name=f"{name}"), row=2, col=1)
        figure.add_trace(go.Scatter(y=np.abs(sig), name=f"{name}"), row=3, col=1)

    eps = 0.0
    for n_taps in [9, 32, 64]:
        sig = compensated_sig(gen_eqz_freq_resp(n_taps, eps=eps)) / norm
        name = f"Wiener {n_taps} taps"
        figure.add_trace(go.Scatter(y=sig.real, name=name), row=1, col=1)
        figure.add_trace(go.Scatter(y=sig.imag, name=name), row=2, col=1)
        figure.add_trace(go.Scatter(y=np.abs(sig), name=name), row=3, col=1)

    if H_ref is not None:
        sig = compensated_sig(H_ref) / norm
        figure.add_trace(go.Scatter(y=sig.real, name="ref"), row=1, col=1)
        figure.add_trace(go.Scatter(y=sig.imag, name="ref"), row=2, col=1)
        figure.add_trace(go.Scatter(y=np.abs(sig), name="ref"), row=3, col=1)

    figure.update_layout(
        yaxis=dict(title="real"),
        yaxis2=dict(title="imag"),
        yaxis3=dict(title="envelope"),
        xaxis3=dict(title="sample"),
        font=dict(size=20),
        margin=dict(l=80, r=20, t=20, b=60),
    )
    fig_dir = os.path.join(_SCRIPT_DIR, "figure")
    os.makedirs(fig_dir, exist_ok=True)
    figure.write_html(os.path.join(fig_dir, "sig_time_domain_compare.html"))


def print_condition_numbers(eps: float = 1e-2):
    desired = np.load(os.path.join(_SCRIPT_DIR, "update_aaf_freq_resp/desired_signal.npy"))
    unwanted = np.load(os.path.join(_SCRIPT_DIR, "update_aaf_freq_resp/unwanted_signal.npy"))
    print(f"{'n_taps':>6}  {'cond(R)':>10}  {'cond(R+εI)':>12}  (ε={eps})")
    for n_taps in [9, 32, 64]:
        n_group = min(len(desired), len(unwanted)) - (n_taps - 1)
        U = unwanted[np.arange(n_group)[:, None] + np.arange(n_taps - 1, -1, -1)[None, :]]
        R = U.conj().T @ U
        cond_before = np.linalg.cond(R)
        R_reg = R + eps * (np.trace(R).real / n_taps) * np.eye(n_taps)
        cond_after = np.linalg.cond(R_reg)
        print(f"{n_taps:>6}  {cond_before:>10.3e}  {cond_after:>12.3e}")


if __name__ == "__main__":
    print_condition_numbers()
    gen_signals()
    H_ref = np.array([
         9.24621582e-01+0.00000000e+00j,  6.36901855e-01-6.73156738e-01j,
        -5.15747070e-02-9.28588867e-01j, -7.12585449e-01-6.02844238e-01j,
        -9.29870605e-01+1.03637695e-01j, -5.64025879e-01+7.47131348e-01j,
         1.54968262e-01+9.22241211e-01j,  7.75207520e-01+5.20568848e-01j,
         9.10400391e-01-2.05017090e-01j,  4.77478027e-01-8.02856445e-01j,
        -2.55249023e-01-9.01062012e-01j, -8.32275391e-01-4.35791016e-01j,
        -8.90991211e-01+3.04992676e-01j, -3.92028809e-01+8.56994629e-01j,
         3.51623535e-01+8.73413086e-01j,  8.73962402e-01+3.45581055e-01j,
         8.51196289e-01-3.95263672e-01j,  2.99804688e-01-8.89526367e-01j,
        -4.39147949e-01-8.31787109e-01j, -9.08325195e-01-2.55737305e-01j,
        -8.13842773e-01+4.83337402e-01j, -2.11242676e-01+9.24438477e-01j,
         5.23986816e-01+7.90527344e-01j,  9.32861328e-01+1.65466309e-01j,
         7.62451172e-01-5.60791016e-01j,  1.19934082e-01-9.39086914e-01j,
        -5.98449707e-01-7.36206055e-01j, -9.49279785e-01-7.49511719e-02j,
        -7.12097168e-01+6.38061523e-01j, -2.95410156e-02+9.58679199e-01j,
         6.74499512e-01+6.83776855e-01j,  9.60083008e-01-1.64184570e-02j,
         6.50024414e-01-7.05688477e-01j, -6.22558594e-02-9.57214355e-01j,
        -7.36816406e-01-6.16210938e-01j, -9.57214355e-01+1.08459473e-01j,
        -5.84106445e-01+7.70446777e-01j,  1.55029297e-01+9.57153320e-01j,
         8.00659180e-01+5.48950195e-01j,  9.49035645e-01-2.00439453e-01j,
         5.08850098e-01-8.23303223e-01j, -2.43957520e-01-9.34448242e-01j,
        -8.43505859e-01-4.68078613e-01j, -9.21386719e-01+2.87048340e-01j,
        -4.29321289e-01+8.65783691e-01j,  3.29772949e-01+9.09423828e-01j,
         8.84887695e-01+3.89831543e-01j,  8.91113281e-01-3.69079590e-01j,
         3.48022461e-01-8.95141602e-01j, -4.03808594e-01-8.66333008e-01j,
        -9.01184082e-01-3.06640625e-01j, -8.42956543e-01+4.36950684e-01j,
        -2.68249512e-01+9.09667969e-01j,  4.69726562e-01+8.23059082e-01j,
         9.17297363e-01+2.31567383e-01j,  8.00170898e-01-4.98596191e-01j,
         1.95068359e-01-9.17480469e-01j, -5.21911621e-01-7.72583008e-01j,
        -9.13269043e-01-1.60034180e-01j, -7.46704102e-01+5.43579102e-01j,
        -1.27868652e-01+9.12780762e-01j,  5.66589355e-01+7.25830078e-01j,
         9.14611816e-01+9.75952148e-02j,  7.04772949e-01-5.87585449e-01j,
         6.82983398e-02-9.11376953e-01j, -6.03637695e-01-6.80053711e-01j,
        -9.03991699e-01-4.01000977e-02j, -6.56066895e-01+6.18591309e-01j,
        -1.31835938e-02+9.00268555e-01j,  6.36657715e-01+6.36718750e-01j,
         9.00756836e-01-1.30004883e-02j,  6.18164062e-01-6.54846191e-01j,
        -3.86962891e-02-8.97888184e-01j, -6.68518066e-01-5.96008301e-01j,
        -8.89892578e-01+6.36596680e-02j, -5.73059082e-01+6.80419922e-01j,
         8.81958008e-02+8.84155273e-01j,  6.95739746e-01+5.53588867e-01j,
         8.83239746e-01-1.12854004e-01j,  5.35583496e-01-7.12402344e-01j,
        -1.36901855e-01-8.80187988e-01j, -7.24548340e-01-5.14709473e-01j,
        -8.71582031e-01+1.59545898e-01j, -4.92614746e-01+7.33825684e-01j,
         1.81640625e-01+8.64257812e-01j,  7.46643066e-01+4.73754883e-01j,
         8.62731934e-01-2.04406738e-01j,  4.57275391e-01-7.62451172e-01j,
        -2.26867676e-01-8.61267090e-01j, -7.75146484e-01-4.39270020e-01j,
        -8.55285645e-01+2.47863770e-01j, -4.19982910e-01+7.85217285e-01j,
         2.69165039e-01+8.50524902e-01j,  8.00231934e-01+4.02954102e-01j,
         8.52600098e-01-2.93395996e-01j,  3.87756348e-01-8.21166992e-01j,
        -3.19763184e-01-8.56201172e-01j, -8.40942383e-01-3.69934082e-01j,
        -8.55041504e-01+3.46557617e-01j, -3.48693848e-01+8.58825684e-01j,
         3.76220703e-01+8.53820801e-01j,  8.83178711e-01+3.26599121e-01j,
         8.58215332e-01-4.12536621e-01j,  3.02551270e-01-9.14428711e-01j,
        -4.53247070e-01-8.60900879e-01j, -9.42199707e-01-2.72033691e-01j,
        -8.54248047e-01+4.94812012e-01j, -2.35168457e-01+9.66613770e-01j,
         5.42114258e-01+8.46740723e-01j,  1.00000000e+00+1.94519043e-01j,
         8.39843750e-01-5.97106934e-01j,  1.45568848e-01-1.02142334e+00j,
        -6.28356934e-01-7.92236328e-01j, -9.54284668e-01-8.33740234e-02j,
        -6.47094727e-01+5.75317383e-01j, -2.15454102e-02+7.36755371e-01j,
         4.12292480e-01+4.11560059e-01j,  4.19860840e-01-1.33666992e-02j,
         1.78161621e-01-2.02270508e-01j, -1.40991211e-02-1.47094727e-01j,
        -5.03540039e-02-3.88793945e-02j, -1.65405273e-02+2.68554688e-03j,
         6.71386719e-04-1.03759766e-03j, -6.10351562e-04-2.50244141e-03j,
         1.15966797e-03+6.71386719e-04j,  3.11279297e-03-9.76562500e-04j,
         6.71386719e-04-1.34277344e-03j,  6.10351562e-04+1.58691406e-03j,
         2.99072266e-03+1.22070312e-03j,  1.52587891e-03-7.32421875e-04j,
        -4.88281250e-04+1.40380859e-03j,  1.52587891e-03+2.80761719e-03j,
         1.22070312e-03+3.05175781e-04j, -2.07519531e-03+1.34277344e-03j,
        -2.44140625e-04+1.15966797e-03j, -9.52148438e-03-1.30004883e-02j,
        -6.04248047e-02-7.38525391e-03j, -1.08215332e-01+9.00878906e-02j,
        -1.55639648e-02+2.55554199e-01j,  2.72460938e-01+2.89978027e-01j,
         5.50598145e-01+1.15966797e-03j,  4.78759766e-01-5.04699707e-01j,
        -4.41894531e-02-8.15307617e-01j, -6.88476562e-01-5.84655762e-01j,
        -9.48608398e-01+1.02844238e-01j, -5.91125488e-01+7.75451660e-01j,
         1.55151367e-01+9.63623047e-01j,  7.98522949e-01+5.47058105e-01j,
         9.37988281e-01-1.99401855e-01j,  4.99389648e-01-8.11828613e-01j,
        -2.42736816e-01-9.19311523e-01j, -8.32153320e-01-4.58068848e-01j,
        -9.04235840e-01+2.85827637e-01j, -4.15527344e-01+8.48876953e-01j,
         3.26416016e-01+8.82385254e-01j,  8.60595703e-01+3.70849609e-01j,
         8.60412598e-01-3.66882324e-01j,  3.27575684e-01-8.76770020e-01j,
        -4.10583496e-01-8.43627930e-01j, -8.96728516e-01-2.84301758e-01j,
        -8.24768066e-01+4.55139160e-01j, -2.37487793e-01+9.11315918e-01j,
         4.97253418e-01+7.98156738e-01j,  9.20227051e-01+1.87683105e-01j,
         7.68981934e-01-5.39428711e-01j,  1.36840820e-01-9.31213379e-01j,
        -5.84594727e-01-7.41455078e-01j, -9.43298340e-01-8.44726562e-02j,
        -7.10083008e-01+6.28845215e-01j, -3.00903320e-02+9.47448730e-01j,
         6.66870117e-01+6.70593262e-01j,  9.42810059e-01-2.48413086e-02j,
         6.27746582e-01-7.01416016e-01j, -7.93457031e-02-9.37500000e-01j,
        -7.36816406e-01-5.86181641e-01j, -9.32800293e-01+1.33483887e-01j,
        -5.43151855e-01+7.69470215e-01j,  1.85729980e-01+9.20959473e-01j,
         7.93334961e-01+4.95666504e-01j,  9.00695801e-01-2.34375000e-01j,
         4.47143555e-01-8.11401367e-01j, -2.80456543e-01-8.80126953e-01j,
        -8.30505371e-01-4.01794434e-01j, -8.62854004e-01+3.25439453e-01j,
        -3.58276367e-01+8.48876953e-01j,  3.67248535e-01+8.42773438e-01j,
         8.60351562e-01+3.13964844e-01j,  8.17138672e-01-4.04357910e-01j,
         2.70263672e-01-8.67004395e-01j, -4.39697266e-01-7.92358398e-01j,
        -8.76831055e-01-2.29187012e-01j, -7.72766113e-01+4.76623535e-01j,
        -1.89636230e-01+8.90075684e-01j,  5.13305664e-01+7.53234863e-01j,
         8.99475098e-01+1.49230957e-01j,  7.28881836e-01-5.46997070e-01j,
         1.07788086e-01-9.04602051e-01j, -5.80627441e-01-7.03369141e-01j,
        -9.12780762e-01-6.56127930e-02j, -6.80908203e-01+6.18774414e-01j,
        -2.16674805e-02+9.25537109e-01j,  6.59301758e-01+6.57348633e-01j,
         9.34631348e-01-2.50854492e-02j,  6.27014160e-01-6.96899414e-01j,
        -7.42797852e-02-9.36401367e-01j, -7.32727051e-01-5.91613770e-01j,
        -9.37133789e-01+1.26037598e-01j, -5.55358887e-01+7.72033691e-01j,
         1.81152344e-01+9.39941406e-01j,  8.12866211e-01+5.16296387e-01j,
         9.37255859e-01-2.38464355e-01j,  4.69970703e-01-8.47595215e-01j,
        -2.95410156e-01-9.23645020e-01j, -8.75427246e-01-4.17602539e-01j,
        -9.04724121e-01+3.52294922e-01j, -3.63525391e-01+9.02770996e-01j,
         4.10644531e-01+8.85864258e-01j,  9.29687500e-01+3.07800293e-01j,
         8.61999512e-01-4.67956543e-01j,  2.48474121e-01-9.48059082e-01j,
        -5.19836426e-01-8.27697754e-01j, -9.55749512e-01-1.86828613e-01j,
        -7.87780762e-01+5.67016602e-01j, -1.25793457e-01+9.60571289e-01j,
         6.13525391e-01+7.48962402e-01j,  9.65637207e-01+6.59790039e-02j,
         7.08862305e-01-6.57470703e-01j,  7.01904297e-03-9.63989258e-01j,
        -6.93603516e-01-6.62841797e-01j, -9.52331543e-01+4.98657227e-02j,
        -6.13647461e-01+7.22900391e-01j,  1.04125977e-01+9.38415527e-01j,
         7.51892090e-01+5.67260742e-01j,  9.27612305e-01-1.56677246e-01j,
         5.22949219e-01-7.80700684e-01j, -2.07092285e-01-9.14306641e-01j,
        -8.03222656e-01-4.76379395e-01j, -8.93737793e-01+2.54211426e-01j,
        -4.28222656e-01+8.19519043e-01j,  2.99011230e-01+8.71520996e-01j,
         8.36914062e-01+3.82202148e-01j,  8.53637695e-01-3.44238281e-01j,
         3.38195801e-01-8.57238770e-01j, -3.89343262e-01-8.35876465e-01j,
        -8.73718262e-01-2.92663574e-01j, -8.12011719e-01+4.31457520e-01j,
        -2.45178223e-01+8.83850098e-01j,  4.71862793e-01+7.84912109e-01j,
         8.94531250e-01+1.97753906e-01j,  7.60620117e-01-5.14282227e-01j,
         1.50390625e-01-9.08935547e-01j, -5.58105469e-01-7.36633301e-01j,
        -9.20227051e-01-1.01501465e-01j, -7.06481934e-01+5.98449707e-01j,
        -5.10253906e-02+9.23522949e-01j,  6.35192871e-01+6.71264648e-01j,
    ])
    H = gen_eqz_freq_resp(n_taps=55, eps=0.0)
    plot_eqz_freq_resp(H)
    plot_sig_freq_resp_compare(H_ref=H_ref)
    plot_sig_time_domain_compare(H_ref=H_ref)

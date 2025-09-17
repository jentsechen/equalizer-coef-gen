import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utility_lib import mkdir
from os import path
import numpy as np
from dataclasses import dataclass, field

imagefolder = "image"


def mkdir_if_a_path(fname):
    if '/' in fname:
        folder = imagefolder + '/' + fname[0:fname.rfind('/')]
    else:
        folder = imagefolder
    if not path.exists(folder):
        mkdir(folder)


def save_waveform_iq(waveform, fname, prefix=None):
    if not path.exists(imagefolder):
        mkdir(imagefolder)
    mkdir_if_a_path(fname)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=np.abs(waveform), name="mag"))
    fig.add_trace(go.Scatter(y=np.real(waveform), name="re"))
    fig.add_trace(go.Scatter(y=np.imag(waveform), name="im"))
    fig.write_html(path.join(imagefolder, prefix + '.' + fname if prefix else fname))


def save_waveform_iq_marker(waveform, fname, prefix=None):
    if not path.exists(imagefolder):
        mkdir(imagefolder)
    mkdir_if_a_path(fname)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=np.abs(waveform), mode='markers', name="mag"))
    fig.add_trace(go.Scatter(y=np.real(waveform), mode='markers', name="re"))
    fig.add_trace(go.Scatter(y=np.imag(waveform), mode='markers', name="im"))
    fig.write_html(path.join(imagefolder, prefix + '.' + fname if prefix else fname))


def save_waveform_cons(waveform, fname, prefix=None, size=1024):
    if not path.exists(imagefolder):
        mkdir(imagefolder)
    mkdir_if_a_path(fname)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.real(waveform), y=np.imag(waveform), mode='markers'))
    fig.update_layout(autosize=True, width=size, height=size)
    fig.write_html(path.join(imagefolder, prefix + '.' + fname if prefix else fname))


def save_waveform_re(waveform, fname, prefix=None, t=None):
    if not path.exists(imagefolder):
        mkdir(imagefolder)
    mkdir_if_a_path(fname)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=np.real(waveform), x=t, name="re"))
    fig.write_html(path.join(imagefolder, prefix + '.' + fname if prefix else fname))


def save_waveform_re_multi(waveforms, fname, prefix=None):
    if not path.exists(imagefolder):
        mkdir(imagefolder)
    mkdir_if_a_path(fname)
    fig = go.Figure()
    for (w, t, n) in waveforms:
        fig.add_trace(go.Scatter(y=np.real(w), x=t, name=n))
    fig.write_html(path.join(imagefolder, prefix + '.' + fname if prefix else fname))


def save_spectrum(waveform, nfft, fname, fs=108e6):
    from scipy.signal import welch
    if not path.exists(imagefolder):
        mkdir(imagefolder)
    mkdir_if_a_path(fname)
    f, pxx = welch(waveform, fs=fs, nperseg=nfft, noverlap=nfft / 2, nfft=nfft, detrend='constant', return_onesided=False, scaling='density', axis=-1, average='mean')
    f = np.fft.fftshift(f)
    pxx = np.fft.fftshift(pxx)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=10 * np.log10(pxx / np.max(pxx))))
    fig.write_html(path.join(imagefolder, fname))

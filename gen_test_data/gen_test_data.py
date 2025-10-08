import struct
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pof
from plotly.subplots import make_subplots
import json

def read_data(port, save_fig_en=False):
    common_path = "./tx_equalizer/waveform/tx_equalizer."
    with open("{}{}.data.json".format(common_path, port), "r", encoding="UTF-8") as f:
        data_j = json.load(f)
    with open("{}{}.valid.json".format(common_path, port), "r", encoding="UTF-8") as f:
        valid_j = json.load(f)
    data = []
    for t in valid_j["timestamp"]:
        if valid_j["buf"][t] == True:
            for i in range(data_j["dim"]["len"]):
                data.append(data_j["buf"][i]["re"][t] + 1j * data_j["buf"][i]["im"][t])
    data = np.array(data[:-5])
    if save_fig_en == True:
        figure = make_subplots(rows=2, cols=1)
        figure.add_trace(go.Scatter(y=data.real), row=1, col=1)
        figure.add_trace(go.Scatter(y=data.imag), row=2, col=1)
        figure.write_html("./{}_data.html".format(port))
    return data

def read_coef():
    common_path = "./tx_equalizer/waveform/tx_equalizer."
    with open("{}inc.coef.json".format(common_path), "r", encoding="UTF-8") as f:
        coef_j = json.load(f)
    coef = []
    for c in coef_j["buf"]:
        coef.append(c["re"][0] + 1j * c["im"][0])
    return np.array(coef)

def pad_zero(data):
    assert len(data) % 5 == 0
    data_zp = []
    for i in range(int(len(data)/5)):
        for j in range(5):
            data_zp.append(data[i*5+j])
        for j in range(3):
            data_zp.append(0)
    return np.array(data_zp)

def save_bin_data(data, file_name):
    data_zp = pad_zero(data)
    interleaved = np.empty(2 * len(data_zp), dtype=np.float32)
    interleaved[1::2] = data_zp.imag
    interleaved[0::2] = data_zp.real
    qntz_data = np.clip(np.round(interleaved * 2**14), -32768, 32767).astype(np.int16)
    with open("./{}.bin".format(file_name), "wb") as f:
        qntz_data.tofile(f)

def load_bin_data(file_name):
    with open("./{}.bin".format(file_name), "rb") as f:
        data = f.read()
    n_bytes = len(data)
    decode_data = []
    for i in range(int(n_bytes/32)):
        for j in range(5):
            s = i*32 + j*4
            # re = float(struct.unpack("h", data[(s+2):(s+4)])[0])
            # im = float(struct.unpack("h", data[s:(s+2)])[0])
            re = float(struct.unpack("h", data[s:(s+2)])[0])
            im = float(struct.unpack("h", data[(s+2):(s+4)])[0])
            decode_data.append((re + 1j*im)/2**14)
    return np.array(decode_data)

def save_coef(coef):
    with open("./coef.json", "w", encoding="UTF-8") as f:
        json.dump({"re": coef.real.tolist(), "im": coef.imag.tolist()}, f)

if __name__ == "__main__":
    ind_data = read_data("ind", True)
    out_data = read_data("out", True)
    coef = read_coef()

    save_bin_data(ind_data, "stimulus")
    save_bin_data(out_data, "capture_memory")
    save_coef(coef)
    ind_data_r = load_bin_data("stimulus")
    out_data_r = load_bin_data("capture_memory")

    figure = make_subplots(rows=2, cols=1)
    figure.add_trace(go.Scatter(y=ind_data.real), row=1, col=1)
    figure.add_trace(go.Scatter(y=ind_data.imag), row=2, col=1)
    figure.add_trace(go.Scatter(y=out_data.real), row=1, col=1)
    figure.add_trace(go.Scatter(y=out_data.imag), row=2, col=1)
    figure.add_trace(go.Scatter(y=ind_data_r.real), row=1, col=1)
    figure.add_trace(go.Scatter(y=ind_data_r.imag), row=2, col=1)
    figure.add_trace(go.Scatter(y=out_data_r.real), row=1, col=1)
    figure.add_trace(go.Scatter(y=out_data_r.imag), row=2, col=1)
    figure.write_html("./result.html")

    figure = make_subplots(rows=2, cols=1)
    calc_golden = np.convolve(ind_data, coef)
    figure.add_trace(go.Scatter(y=calc_golden.real), row=1, col=1)
    figure.add_trace(go.Scatter(y=calc_golden.imag), row=2, col=1)
    figure.write_html("./calc_golden.html")

    print("DONE")


import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pow_amp_non_lin_rapp import PowAmpNonLinRapp

if __name__ == "__main__":
    pow_amp_non_lin_poly = PowAmpNonLinRapp()

    in_pow_dbm = np.arange(-40, 10, 0.01)
    in_volt_pk = []
    for x in in_pow_dbm:
        in_volt_pk.append(pow_amp_non_lin_poly.power_dbm_to_voltage_peak(x))
    in_volt_pk = np.array(in_volt_pk)

    out_volt_pk_non_lin = pow_amp_non_lin_poly.apply(in_volt_pk)
    out_pow_dbm_non_lin = []
    for y in out_volt_pk_non_lin:
        out_pow_dbm_non_lin.append(pow_amp_non_lin_poly.voltage_peak_to_power_dbm(abs(y)))
    out_pow_dbm_non_lin = np.array(out_pow_dbm_non_lin)

    out_pow_dbm = in_pow_dbm + pow_amp_non_lin_poly.gain_db

    idx = (np.abs(out_pow_dbm_non_lin - pow_amp_non_lin_poly.op1db_dbm)).argmin()
    print("OP1dB: {}".format(out_pow_dbm_non_lin[idx]))
    print("OP1dB+1dB: {}".format(out_pow_dbm[idx]))
    print("Diff.: {}".format(out_pow_dbm[idx] - out_pow_dbm_non_lin[idx]))
    print("IP1dB: {}".format(in_pow_dbm[idx]))

    figure = make_subplots(rows=1, cols=1)
    figure.add_trace(go.Scatter(
        x=in_pow_dbm, 
        y=(out_pow_dbm), 
        line=dict(color="blue", width=2),
        name="linear"), row=1, col=1)
    figure.add_trace(go.Scatter(
        x=in_pow_dbm, 
        y=out_pow_dbm_non_lin, 
        line=dict(color="red", width=2),
        name="non-linear"), row=1, col=1)
    figure.add_hline(y=pow_amp_non_lin_poly.p_sat_dbm, line_dash="dash", annotation_text="Psat")
    figure.add_hline(y=pow_amp_non_lin_poly.op1db_dbm, line_dash="dash", annotation_text="OP1dB")
    figure.add_hline(y=pow_amp_non_lin_poly.op1db_dbm+1, line_dash="dash", annotation_text="OP1dB+1dB")
    figure.add_vline(x=in_pow_dbm[idx], line_dash="dash", annotation_text="IP1dB")
    figure.update_layout(
        xaxis_title="Input Power (dBm)",
        yaxis_title="Output Power (dBm)",
        font=dict(size=30)
    )
    figure.write_html("in_out_pow.html")

    print("DONE")
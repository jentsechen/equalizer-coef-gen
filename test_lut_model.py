import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import json
from pow_amp_non_lin_lut import PowAmpNonLinLut

if __name__ == "__main__":
    lut_model = PowAmpNonLinLut()

    in_pow_dbm = np.arange(-30, 10, 0.01)
    in_volt_pk = []
    for x in in_pow_dbm:
        in_volt_pk.append(lut_model.power_dbm_to_voltage_peak(x))
    in_volt_pk = np.array(in_volt_pk)

    out_pow_dbm_non_lin, out_phase_rad = [], []
    for x in in_volt_pk:
        y = lut_model.apply(x)
        out_pow_dbm_non_lin.append(lut_model.voltage_peak_to_power_dbm(abs(y)))
        out_phase_rad.append(np.angle(y, deg=True))
    out_pow_dbm_non_lin = np.array(out_pow_dbm_non_lin)
    out_phase_rad = np.array(out_phase_rad)

    figure = make_subplots(rows=2, cols=1)
    figure.add_trace(go.Scatter(
        x=in_pow_dbm, 
        y=out_pow_dbm_non_lin, 
        line=dict(color="blue", width=2),
        name=""), row=1, col=1)
    figure.add_trace(go.Scatter(
        x=in_pow_dbm, 
        y=out_phase_rad, 
        line=dict(color="blue", width=2),
        name=""), row=2, col=1)
    # figure.add_hline(y=pow_amp_non_lin_poly.p_sat_dbm, line_dash="dash", annotation_text="Psat")
    # figure.add_hline(y=pow_amp_non_lin_poly.op1db_dbm, line_dash="dash", annotation_text="OP1dB")
    # figure.add_hline(y=pow_amp_non_lin_poly.op1db_dbm+1, line_dash="dash", annotation_text="OP1dB+1dB")
    # figure.add_vline(x=in_pow_dbm[idx], line_dash="dash", annotation_text="IP1dB")
    # figure.update_layout(
    #     xaxis_title="Input Power (dBm)",
    #     yaxis_title="Output Power (dBm)",
    #     font=dict(size=30)
    # )
    figure.write_html("amam_ampm.html")

    print("DONE")

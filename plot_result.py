import numpy as np
import json
import plotly.graph_objs as go
import plotly.offline as pof
from plotly.subplots import make_subplots

if __name__ == "__main__":
    with open("result.json", "r", encoding="UTF-8") as f:
        result = json.load(f)
    
    figure = make_subplots(rows=2, cols=1)
    for i in range(len(result["pslr_db"])):
        figure.add_trace(go.Scatter(x=np.arange(1, len(result["pslr_db"])+1), 
                                    y=result["pslr_db"][i][1:], line=dict(width=2)), row=1, col=1)
        figure.add_trace(go.Scatter(x=np.arange(1, len(result["pslr_db"])+1), 
                                    y=result["irw_m"][i][1:], line=dict(width=2)), row=2, col=1)
    # figure.add_hline(y=13.263395324071894, line_dash="dash", row=1, col=1)
    # figure.add_hline(y=0.2052, line_dash="dash", row=2, col=1)
    figure.update_layout(
        xaxis=dict(title="i of v_i (i=1 for initial estimation by Wiener-Hopf algorithm)"),       
        yaxis=dict(title="PSLR (dB)"),
        xaxis2=dict(title="i of v_i (i=1 for initial estimation by Wiener-Hopf algorithm)"),
        yaxis2=dict(title="IRW (m)"),
        font=dict(size=25)
    )
    figure.write_html("result.html")

    print("DONE")

    # iof\mathbf
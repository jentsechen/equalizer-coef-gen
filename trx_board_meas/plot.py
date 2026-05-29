from pathlib import Path

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from s2p import _DATA_DIR, _FIGURE_DIR, _load_decimated

_COLORS = px.colors.qualitative.Plotly


def _group(stem: str):
    """'tx_ch0_h2' → 'tx_ch0'"""
    return stem.rsplit("_", 1)[0]


def _plot(files: list, out_name: str):
    _FIGURE_DIR.mkdir(exist_ok=True)

    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.08,
    )

    for i, path in enumerate(files):
        color = _COLORS[i % len(_COLORS)]
        group = _group(path.stem)
        freq, gain, phase = _load_decimated(path)

        fig.add_trace(
            go.Scatter(x=freq, y=gain, name=path.stem, legendgroup=group,
                       line=dict(color=color)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=freq, y=phase, name=path.stem, legendgroup=group,
                       showlegend=False, line=dict(color=color)),
            row=2, col=1,
        )

    fig.update_xaxes(title_text="Frequency (GHz)")
    fig.update_yaxes(title_text="Gain (dB)", row=1, col=1)
    fig.update_yaxes(title_text="Phase (deg)", row=2, col=1)
    fig.update_layout(
        font=dict(size=20),
        margin=dict(l=80, r=20, t=60, b=60),
        legend=dict(groupclick="toggleitem"),
    )

    out = _FIGURE_DIR / out_name
    fig.write_html(out)
    print(f"saved → {out}")


def plot_tx_stx():
    files = sorted(_DATA_DIR.glob("tx_*.s2p")) + sorted(_DATA_DIR.glob("stx_*.s2p"))
    _plot(files, out_name="tx_stx.html")


def plot_rx_srx():
    files = sorted(_DATA_DIR.glob("rx_*.s2p")) + sorted(_DATA_DIR.glob("srx_*.s2p"))
    _plot(files, out_name="rx_srx.html")


def main():
    plot_tx_stx()
    plot_rx_srx()


if __name__ == "__main__":
    main()

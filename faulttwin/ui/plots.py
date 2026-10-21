"""Plotly figures. One fixed colour per phase everywhere; units on every axis; spectra in rad/s."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from faulttwin.features import clarke
from faulttwin.ui.views import PlantView

PHASES = ("A", "B", "C")
PHASE_COLORS = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73"}   # colour-blind safe
TWIN_DASH = "dash"
# The empty template with a few explicit styles: a full template is copied into every figure on every update
# (measured: 9.1 ms and 51 KB per plant update with plotly_white, 4.2 ms and 31 KB with this).
TEMPLATE = pio.templates["none"]
_AXIS = dict(gridcolor="#E6E8EB", zeroline=False, showline=True, linecolor="#C9CED6", ticks="outside",
             tickcolor="#C9CED6", automargin=True)
_BASE = dict(margin=dict(l=52, r=16, t=64, b=44), height=340, template=TEMPLATE,
             font=dict(size=12, color="#1F2933"), paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
             xaxis=_AXIS, yaxis=_AXIS,
             legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", font=dict(size=11)))


def _title(text: str) -> dict[str, Any]:
    return dict(text=text, x=0, xanchor="left", y=0.98, yanchor="top", font=dict(size=14))


def currents_figure(v: PlantView) -> go.Figure:
    fig = go.Figure()
    t_ms = (v.t_s - (v.t_s[0] if len(v.t_s) else 0.0)) * 1e3
    for k, ph in enumerate(PHASES):
        fig.add_trace(go.Scatter(x=t_ms, y=v.i_A[:, k] if len(v.i_A) else [], name=f"i_{ph.lower()}",
                                 line=dict(color=PHASE_COLORS[ph], width=1.6)))
    fig.update_layout(title=_title("Phase currents (last 3 cycles)"), xaxis_title="t (ms)", yaxis_title="i (A)",
                      **_BASE)
    return fig


_THETA = np.linspace(0, 2 * math.pi, 120)


def _park_data(v: PlantView, limit_A: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    a, b = clarke(v.i_A) if len(v.i_A) else (np.zeros(0), np.zeros(0))
    if len(a):
        # Fixed axes in normal use, so a deforming vector is easy to see; they grow only when the trace needs it.
        limit_A = max(limit_A, 1.1 * float(max(np.abs(a).max(), np.abs(b).max())))
    r = v.features["i_s_A"] if v.features else 0.0
    return a, b, r * np.cos(_THETA), r * np.sin(_THETA), limit_A


def park_figure(v: PlantView, limit_A: float = 35.0) -> go.Figure:
    a, b, cx, cy, limit_A = _park_data(v, limit_A)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a, y=b, mode="lines", name="measured", line=dict(color="#444444", width=1.6)))
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", name="|i_s| circle",
                             line=dict(color="#BBBBBB", width=1, dash="dot")))
    fig.update_layout(title=_title("Park's vector"), xaxis_title="i_α (A)", yaxis_title="i_β (A)",
                      **dict(_BASE, height=340))
    fig.update_xaxes(range=[-limit_A, limit_A], constrain="domain")
    fig.update_yaxes(range=[-limit_A, limit_A], scaleanchor="x", scaleratio=1, constrain="domain")
    return fig


SPECTRUM_LABELS = ["0", "ω₁ = 314", "2ω₁ = 628", "3ω₁ = 942"]


def _spectrum_rows(v: PlantView) -> list[list[float]]:
    """Per phase, the amplitudes at 0, ω₁, 2ω₁, 3ω₁ (zeros until the monitor has features)."""
    spectrum = (v.features or {}).get("spectrum_A", {})
    bins = list(spectrum)[: len(SPECTRUM_LABELS)]
    return [[spectrum[b][k] for b in bins] + [0.0] * (len(SPECTRUM_LABELS) - len(bins)) for k in range(3)]


def spectrum_figure(v: PlantView) -> go.Figure:
    fig = go.Figure()
    for k, ph in enumerate(PHASES):
        fig.add_trace(go.Bar(x=SPECTRUM_LABELS, y=_spectrum_rows(v)[k], name=f"phase {ph}",
                             marker_color=PHASE_COLORS[ph]))
    fig.update_layout(title=_title("ω-spectrum"), xaxis_title="ω (rad/s)", yaxis_title="amplitude (A)",
                      barmode="group", **_BASE)
    return fig


class PlantFigures:
    """The three plant figures of one session, built once and then updated in place.

    Building a Plotly figure costs about 15 ms (property validation); setting the trace data of an
    existing one costs about 2 ms, so a plant update takes ~6 ms instead of ~55 ms.
    """

    def __init__(self) -> None:
        self.currents: go.Figure | None = None
        self.park: go.Figure | None = None
        self.spectrum: go.Figure | None = None

    def update(self, v: PlantView) -> tuple[go.Figure, go.Figure, go.Figure]:
        if self.currents is None:
            self.currents, self.park, self.spectrum = currents_figure(v), park_figure(v), spectrum_figure(v)
            return self.currents, self.park, self.spectrum
        t_ms = (v.t_s - (v.t_s[0] if len(v.t_s) else 0.0)) * 1e3
        with self.currents.batch_update():
            for k in range(3):
                self.currents.data[k].x = t_ms
                self.currents.data[k].y = v.i_A[:, k] if len(v.i_A) else []
        a, b, cx, cy, limit = _park_data(v, 35.0)
        with self.park.batch_update():
            self.park.data[0].x, self.park.data[0].y = a, b
            self.park.data[1].x, self.park.data[1].y = cx, cy
            self.park.layout.xaxis.range = [-limit, limit]
            self.park.layout.yaxis.range = [-limit, limit]
        rows = _spectrum_rows(v)
        with self.spectrum.batch_update():
            for k in range(3):
                self.spectrum.data[k].y = rows[k]
        return self.currents, self.park, self.spectrum


def ranking_figure(ranked: list[dict[str, Any]]) -> go.Figure:
    rows = list(reversed(ranked[:6]))
    colors = ["#C0392B" if r["converter_fault"] else "#7F8C8D" for r in rows]
    names = [f"{r['hypothesis']} (fault)" if r["converter_fault"] else r["hypothesis"] for r in rows]
    fig = go.Figure(go.Bar(x=[r["match_error_pct"] for r in rows], y=names, orientation="h",
                           marker_color=colors, text=[f"{r['match_error_pct']:.2f} %" for r in rows],
                           textposition="outside", cliponaxis=False, showlegend=False))
    layout = dict(_BASE, height=270, margin=dict(l=52, r=64, t=48, b=44))
    fig.update_layout(title=_title("Twin match error (lower is better)"), xaxis_title="match error (% of |i_s|)",
                      **layout)
    return fig


def overlay_figure(overlays: dict[str, Any], hypothesis: str) -> go.Figure:
    t = np.asarray(overlays["t_s"], dtype=float)
    t_ms = (t - t[0]) * 1e3
    measured = np.asarray(overlays["measured_A"], dtype=float)
    predicted = np.asarray(overlays["predicted_A"][hypothesis], dtype=float)
    fig = go.Figure()
    for k, ph in enumerate(PHASES):
        fig.add_trace(go.Scatter(x=t_ms, y=measured[:, k], name=f"measured {ph}",
                                 line=dict(color=PHASE_COLORS[ph], width=1.4)))
    for k, ph in enumerate(PHASES):
        fig.add_trace(go.Scatter(x=t_ms, y=predicted[:, k], name="twin (dashed, same colours)", legendgroup="twin",
                                 showlegend=k == 0, line=dict(color=PHASE_COLORS[ph], width=1.4, dash=TWIN_DASH)))
    fig.update_layout(title=_title(f"Measured (solid) vs twin '{hypothesis}' (dashed)"), xaxis_title="t (ms)",
                      yaxis_title="i (A)", **_BASE)
    return fig

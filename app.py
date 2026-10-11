"""WellSentinel — live early-warning console.

Reads a model bundle (see the ML core repo, ``scripts/export_bundle.py``) and replays a
held-out well: the operator sees the sensor traces, a risk curve, the lead time the model
would have given, a plain explanation and a recommended action — behind a review gate.

The bundle contains precomputed per-window risk, so this app only reads Parquet and never
loads the model itself. That keeps the demo light and the number honest: every instance in
``bundle/demo`` was held out of training.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BUNDLE = Path(os.environ.get("WELLSENTINEL_BUNDLE", "bundle"))
HORIZON_HOURS = 6.0
ALERT_THRESHOLD = 0.5

st.set_page_config(page_title="WellSentinel", page_icon="🛢️", layout="wide")

# -- Recommended actions per event type ------------------------------------------
ACTIONS: dict[str, tuple[str, str]] = {
    "ABRUPT_INCREASE_OF_BSW": (
        "Monitor basic sediment & water; check downstream separation.",
        "Rising BSW means more water is being produced. Route to test separator and watch "
        "the water cut trend.",
    ),
    "SPURIOUS_CLOSURE_OF_DHSV": (
        "Verify downhole safety valve status; do not override automatically.",
        "A spurious DHSV closure cuts production with no physical cause. Confirm at the "
        "panel and follow the restart procedure.",
    ),
    "SEVERE_SLUGGING": (
        "Review choke opening and gas-lift rate for the slugging cycle.",
        "Slugging appears as a repeating pressure oscillation. A small choke change often "
        "damps the cycle.",
    ),
    "FLOW_INSTABILITY": (
        "Hold the current choke; watch the pressure envelope for stabilisation.",
        "Flow instability shows as widening pressure swings. Hold steady and let it settle "
        "before any change.",
    ),
    "RAPID_PRODUCTIVITY_LOSS": (
        "Check for a restriction; review choke and flowline differential.",
        "A fast drop in productivity suggests a growing restriction or a partial blockage.",
    ),
    "QUICK_RESTRICTION_IN_PCK": (
        "Inspect the production choke for a restriction.",
        "A quick choke restriction raises the upstream-downstream differential sharply.",
    ),
    "SCALING_IN_PCK": (
        "Flag for scale inspection; trend the choke differential over days.",
        "Scaling builds slowly. The differential creeps up over a long horizon.",
    ),
    "HYDRATE_IN_PRODUCTION_LINE": (
        "Raise inhibitor dose / check line temperature immediately.",
        "Hydrate formation is pressure and temperature driven. Check the downstream "
        "temperature trend and inhibitor availability.",
    ),
    "HYDRATE_IN_SERVICE_LINE": (
        "Check the service line inhibitor and temperature.",
        "Hydrate risk on the service line. Confirm inhibitor dosing before continuing.",
    ),
}


@st.cache_data(show_spinner=False)
def load_manifest() -> dict:
    path = BUNDLE / "manifest.json"
    if not path.exists():
        return {"metrics": {}, "instances": []}
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def load_instance(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(BUNDLE / "demo" / f"{name}.parquet")
    pred = pd.read_parquet(BUNDLE / "demo" / f"{name}.pred.parquet")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    pred["window_start"] = pd.to_datetime(pred["window_start"])
    return raw, pred


def risk_figure(pred: pd.DataFrame, lead_hours: float | None) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=ALERT_THRESHOLD, y1=1.0, fillcolor="rgba(220,50,47,0.10)", line_width=0)
    fig.add_trace(
        go.Scatter(
            x=pred["window_start"],
            y=pred["risk"],
            name="Risk",
            line=dict(color="#d62728", width=2),
        )
    )
    fig.add_hline(
        y=ALERT_THRESHOLD,
        line_dash="dot",
        line_color="#888",
        annotation_text=f"alert ≥ {ALERT_THRESHOLD}",
        annotation_position="bottom right",
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="P(event within 6 h)", range=[0, 1]),
        xaxis_title=None,
        showlegend=False,
    )
    return fig


def traces_figure(raw: pd.DataFrame, pred: pd.DataFrame) -> go.Figure:
    numeric = [
        c
        for c in raw.columns
        if c not in {"timestamp", "class", "state"} and raw[c].notna().any() and raw[c].nunique() > 1
    ][:6]
    fig = go.Figure()
    for column in numeric:
        fig.add_trace(go.Scatter(x=raw["timestamp"], y=raw[column], name=column, line=dict(width=1.2)))
    if "risk" in pred and (pred["risk"] >= ALERT_THRESHOLD).any():
        first = pred.loc[pred["risk"] >= ALERT_THRESHOLD, "window_start"].iloc[0]
        fig.add_vline(x=first, line_dash="dash", line_color="#d62728", annotation_text="first alert")
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=-0.2),
        xaxis_title=None,
    )
    return fig


def main() -> None:
    manifest = load_manifest()
    metrics = manifest.get("metrics", {})
    instances = manifest.get("instances", [])

    st.title("🛢️ WellSentinel")
    st.caption("Early warning for undesirable events in oil wells · held-out wells only, nothing actioned automatically")

    if not instances:
        st.error(
            "No demo bundle found. Run `python scripts/export_bundle.py` in the ML core repo and "
            f"point `WELLSENTINEL_BUNDLE` at it (currently `{BUNDLE}`)."
        )
        return

    with st.sidebar:
        st.header("Well under watch")
        names = [i["name"] for i in instances]
        choice = st.selectbox("Held-out instance", names, index=0)
        info = next(i for i in instances if i["name"] == choice)
        st.metric("Event type", info["event"])
        st.metric("Well", info["well_id"])
        if info.get("normal_lead_hours") is not None:
            st.metric("Normal lead-in before event", f"{info['normal_lead_hours']:.1f} h")
        st.metric("Instance length", f"{info['span_hours']:.1f} h")
        st.divider()
        st.caption("Held out of training")
        st.write(f"ROC AUC **{metrics.get('roc_auc', '—')}**")
        st.write(f"Avg precision **{metrics.get('average_precision', '—')}**")
        st.write(f"Test wells **{', '.join(metrics.get('test_wells', []))}**")

    raw, pred = load_instance(choice)
    alerted = pred[pred["risk"] >= ALERT_THRESHOLD]
    max_risk = float(pred["risk"].max()) if len(pred) else 0.0
    lead_hours = None
    if len(alerted) and pred["seconds_to_onset"].notna().any():
        lead_hours = float(alerted["seconds_to_onset"].max()) / 3600.0

    left, mid, right = st.columns(3)
    left.metric("Peak risk (6 h horizon)", f"{max_risk*100:.0f}%", delta="above alert" if max_risk >= ALERT_THRESHOLD else "below alert")
    mid.metric("Lead time at first alert", f"{lead_hours:.1f} h" if lead_hours else "no alert")
    right.metric("Alert windows", f"{len(alerted)} / {len(pred)}")

    st.subheader("Risk to the event horizon")
    st.plotly_chart(risk_figure(pred, lead_hours), use_container_width=True)

    st.subheader("Sensor traces")
    st.plotly_chart(traces_figure(raw, pred), use_container_width=True)

    st.subheader("What the operator would see")
    action, why = ACTIONS.get(info["event"], ("Review the well.", "Event type not mapped."))
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"**Recommended action.** {action}")
        st.markdown(f"**Why.** {why}")
        if lead_hours:
            st.markdown(
                f"**Warning.** The model crossed the alert line about **{lead_hours:.1f} h** "
                "before the recorded onset."
            )
    with c2:
        st.markdown("**Human review gate**")
        st.radio(
            "Decision",
            ["Pending review", "Acknowledge — monitor", "Escalate to field engineer", "Dismiss"],
            index=0,
            key=f"gate-{choice}",
        )
        st.caption("Nothing is actioned automatically. This is a decision aid, not a controller.")

    with st.expander("How to read this / honesty notes"):
        st.markdown(
            "- **Risk** is the model's probability that an undesirable event starts within "
            f"**{HORIZON_HOURS:.0f} hours**.\n"
            "- Windows record the horizon *from the window end*, so this is prediction, not detection.\n"
            "- The instance was **held out**: its well never appeared in training.\n"
            "- Current scores are a first pass and are **weak** (ROC AUC near 0.6). Read them as "
            "signal-present, not production-ready.\n"
            "- Simulated wells are used in training only; nothing here is graded on the simulator."
        )


if __name__ == "__main__":
    main()

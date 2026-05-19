import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Early Fault Diagnosis Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASSES = ["Normal", "Fault_8", "Fault_10", "Fault_13", "Fault_17", "Fault_18", "Fault_20"]
VARIABLES = [
    "temperature",
    "pressure",
    "flow_rate",
    "liquid_level",
    "composition",
    "valve_opening",
    "feed_rate",
]

VARIABLE_META = {
    "temperature": {"label": "Temperature", "unit": "°C", "min": 20.0, "max": 200.0, "normal": 95.0},
    "pressure": {"label": "Pressure", "unit": "bar", "min": 1.0, "max": 50.0, "normal": 18.0},
    "flow_rate": {"label": "Flow Rate", "unit": "kg/h", "min": 20.0, "max": 200.0, "normal": 100.0},
    "liquid_level": {"label": "Liquid Level", "unit": "%", "min": 10.0, "max": 100.0, "normal": 55.0},
    "composition": {"label": "Composition", "unit": "%", "min": 0.0, "max": 100.0, "normal": 48.0},
    "valve_opening": {"label": "Valve Opening", "unit": "%", "min": 0.0, "max": 100.0, "normal": 62.0},
    "feed_rate": {"label": "Feed Rate", "unit": "kg/h", "min": 20.0, "max": 220.0, "normal": 110.0},
}

FAULT_PROFILES = {
    "Normal": {"temperature": 0.0, "pressure": 0.0, "flow_rate": 0.0, "liquid_level": 0.0, "composition": 0.0, "valve_opening": 0.0, "feed_rate": 0.0},
    "Fault_8": {"temperature": 0.10, "pressure": -0.05, "flow_rate": 0.12, "liquid_level": 0.06, "composition": 0.04, "valve_opening": 0.15, "feed_rate": 0.10},
    "Fault_10": {"temperature": 0.18, "pressure": 0.12, "flow_rate": -0.10, "liquid_level": -0.06, "composition": 0.16, "valve_opening": -0.04, "feed_rate": 0.08},
    "Fault_13": {"temperature": 0.22, "pressure": 0.28, "flow_rate": -0.12, "liquid_level": 0.04, "composition": 0.10, "valve_opening": -0.18, "feed_rate": 0.06},
    "Fault_17": {"temperature": -0.06, "pressure": 0.16, "flow_rate": 0.10, "liquid_level": 0.20, "composition": -0.08, "valve_opening": 0.12, "feed_rate": 0.05},
    "Fault_18": {"temperature": 0.26, "pressure": 0.08, "flow_rate": -0.22, "liquid_level": -0.10, "composition": 0.06, "valve_opening": 0.05, "feed_rate": -0.18},
    "Fault_20": {"temperature": -0.04, "pressure": 0.24, "flow_rate": -0.08, "liquid_level": 0.22, "composition": -0.16, "valve_opening": -0.10, "feed_rate": 0.12},
}

FAULT_DESCRIPTIONS = {
    "Normal": "Process is operating near nominal conditions.",
    "Fault_8": "Mild actuator/process mismatch with elevated flow and valve movement.",
    "Fault_10": "Thermal-composition drift with moderate pressure rise.",
    "Fault_13": "High-risk thermal and pressure escalation with valve restriction.",
    "Fault_17": "Inventory/level-related deviation with elevated pressure and flow.",
    "Fault_18": "Early feed-flow disturbance with strong throughput reduction signature.",
    "Fault_20": "Pressure-level coupling fault with composition drop.",
}

@dataclass
class DiagnosisResult:
    predicted_fault: str
    confidence: float
    conventional_time: float
    ai_time: float
    time_saved: float
    severity: float
    probabilities: Dict[str, float]
    risk_level: str

def get_normalized_deviation(values: Dict[str, float]) -> Dict[str, float]:
    deviations = {}
    for key, value in values.items():
        meta = VARIABLE_META[key]
        span = meta["max"] - meta["min"]
        deviations[key] = (value - meta["normal"]) / span
    return deviations

def severity_score(values: Dict[str, float]) -> float:
    dev = get_normalized_deviation(values)
    score = float(sum(abs(v) for v in dev.values()) / len(dev))
    return min(1.0, score * 6.0)

def profile_similarity(values: Dict[str, float], fault_name: str) -> float:
    dev = get_normalized_deviation(values)
    profile = FAULT_PROFILES[fault_name]
    dist = 0.0
    for key in VARIABLES:
        dist += (dev[key] - profile[key]) ** 2
    dist = math.sqrt(dist / len(VARIABLES))
    return math.exp(-5.0 * dist)

def compute_probabilities(values: Dict[str, float]) -> Dict[str, float]:
    raw = {}
    for fault in CLASSES:
        raw[fault] = profile_similarity(values, fault)
    temp = 1.6
    exp_scores = {k: math.exp(v * temp) for k, v in raw.items()}
    total = sum(exp_scores.values())
    probs = {k: v / total for k, v in exp_scores.items()}
    return probs

def estimate_detection_time(confidence: float, severity: float) -> Tuple[float, float, float]:
    conventional = 20.0
    reduction = 2.0 + severity * 5.5 + confidence * 3.5
    ai_time = max(5.0, conventional - reduction)
    saved = conventional - ai_time
    return conventional, ai_time, saved

def risk_from_severity(severity: float, predicted_fault: str) -> str:
    if predicted_fault == "Normal" and severity < 0.18:
        return "Low"
    if severity < 0.32:
        return "Moderate"
    if severity < 0.52:
        return "High"
    return "Critical"

def diagnose(values: Dict[str, float]) -> DiagnosisResult:
    probs = compute_probabilities(values)
    predicted = max(probs, key=probs.get)
    confidence = probs[predicted]
    severity = severity_score(values)

    if severity < 0.12 and probs["Normal"] > 0.24:
        predicted = "Normal"
        confidence = max(confidence, probs["Normal"])

    conventional, ai_time, saved = estimate_detection_time(confidence, severity)
    risk = risk_from_severity(severity, predicted)

    return DiagnosisResult(
        predicted_fault=predicted,
        confidence=confidence,
        conventional_time=conventional,
        ai_time=ai_time,
        time_saved=saved,
        severity=severity,
        probabilities=probs,
        risk_level=risk,
    )

def build_trend(values: Dict[str, float], steps: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(steps):
        progress = i / max(1, steps - 1)
        row = {"Minute": i}
        for key in VARIABLES:
            normal = VARIABLE_META[key]["normal"]
            current = values[key]
            simulated = normal + (current - normal) * progress + rng.normal(0, 0.01 * (VARIABLE_META[key]["max"] - VARIABLE_META[key]["min"]))
            row[key] = simulated
        rows.append(row)
    return pd.DataFrame(rows)

def format_fault_label(fault: str) -> str:
    return fault.replace("_", " ")

st.sidebar.title("Process Input Panel")
st.sidebar.caption("Manually change the operating variables and run AI-based early fault diagnosis.")

preset = st.sidebar.selectbox(
    "Quick scenario",
    ["Custom", "Nominal Operation", "Pressure Surge", "Flow Disturbance", "Valve Issue", "Composition Drift"],
)

base_values = {k: VARIABLE_META[k]["normal"] for k in VARIABLES}

if preset == "Pressure Surge":
    base_values.update({"pressure": 31.0, "temperature": 140.0, "valve_opening": 38.0})
elif preset == "Flow Disturbance":
    base_values.update({"flow_rate": 63.0, "feed_rate": 72.0, "temperature": 132.0})
elif preset == "Valve Issue":
    base_values.update({"valve_opening": 28.0, "pressure": 28.0, "liquid_level": 72.0})
elif preset == "Composition Drift":
    base_values.update({"composition": 66.0, "temperature": 128.0, "flow_rate": 82.0})

values: Dict[str, float] = {}
for key in VARIABLES:
    meta = VARIABLE_META[key]
    values[key] = st.sidebar.slider(
        f"{meta['label']} ({meta['unit']})",
        min_value=float(meta["min"]),
        max_value=float(meta["max"]),
        value=float(base_values[key]),
        step=float((meta["max"] - meta["min"]) / 100.0),
    )

run_button = st.sidebar.button("Run Diagnosis", type="primary", use_container_width=True)

if "last_values" not in st.session_state:
    st.session_state.last_values = values.copy()

if run_button:
    st.session_state.last_values = values.copy()

active_values = st.session_state.last_values
result = diagnose(active_values)
trend_df = build_trend(active_values)

st.title("Interactive Early Fault Diagnosis Demo")
st.markdown(
    "Local demo for chemical-process monitoring using a **CNN-BiGRU-inspired early fault diagnosis engine**. "
    "Users can manually vary process inputs such as temperature, pressure, flow rate, composition, and valve opening."
)

metric_cols = st.columns(5)
metric_cols[0].metric("Predicted Fault", format_fault_label(result.predicted_fault))
metric_cols[1].metric("Confidence", f"{result.confidence * 100:.1f}%")
metric_cols[2].metric("AI Detection Time", f"{result.ai_time:.1f} min")
metric_cols[3].metric("Conventional Detection", f"{result.conventional_time:.1f} min")
metric_cols[4].metric("Time Saved", f"{result.time_saved:.1f} min")

left, right = st.columns([1.2, 1.0])

with left:
    st.subheader("Current Diagnosis")
    if result.predicted_fault == "Normal":
        st.success(f"Condition: {format_fault_label(result.predicted_fault)}")
    elif result.risk_level in {"Moderate", "High"}:
        st.warning(f"Condition: {format_fault_label(result.predicted_fault)} | Risk: {result.risk_level}")
    else:
        st.error(f"Condition: {format_fault_label(result.predicted_fault)} | Risk: {result.risk_level}")

    st.write(FAULT_DESCRIPTIONS[result.predicted_fault])
    st.progress(min(100, int(result.confidence * 100)), text=f"Model confidence: {result.confidence * 100:.1f}%")

    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame(
        {
            "Fault Class": [format_fault_label(c) for c in CLASSES],
            "Probability": [result.probabilities[c] for c in CLASSES],
        }
    ).sort_values("Probability", ascending=False)
    st.bar_chart(prob_df.set_index("Fault Class"))

    st.subheader("Input Variables")
    input_df = pd.DataFrame(
        {
            "Variable": [VARIABLE_META[k]["label"] for k in VARIABLES],
            "Value": [active_values[k] for k in VARIABLES],
            "Unit": [VARIABLE_META[k]["unit"] for k in VARIABLES],
            "Normal": [VARIABLE_META[k]["normal"] for k in VARIABLES],
        }
    )
    st.dataframe(input_df, use_container_width=True, hide_index=True)

with right:
    st.subheader("Detection Timeline")
    timeline_df = pd.DataFrame(
        {
            "Method": ["AI Model", "Conventional Sensor/Alarm"],
            "Detection Time (min)": [result.ai_time, result.conventional_time],
        }
    )
    st.bar_chart(timeline_df.set_index("Method"))

    st.subheader("Severity and Early Warning")
    st.metric("Severity Index", f"{result.severity * 100:.1f}%")
    if result.predicted_fault == "Normal":
        st.info("No major abnormality detected. Process remains close to nominal behavior.")
    else:
        st.markdown(
            f"**Early warning:** The model indicates **{format_fault_label(result.predicted_fault)}** and could raise the alert about **{result.time_saved:.1f} minutes earlier** than conventional monitoring."
        )

    st.subheader("Recommended Action")
    action_map = {
        "Normal": "Continue normal monitoring and keep process variables near nominal setpoints.",
        "Fault_8": "Inspect actuator response and check valve/flow synchronization.",
        "Fault_10": "Review composition control loop and thermal stability of the unit.",
        "Fault_13": "Urgent: inspect pressure rise, temperature escalation, and valve restriction.",
        "Fault_17": "Check vessel inventory, level control, and possible pressure accumulation.",
        "Fault_18": "Inspect feed throughput path and upstream flow disturbances immediately.",
        "Fault_20": "Review pressure-level coupling and composition control performance.",
    }
    st.write(action_map[result.predicted_fault])

st.subheader("Simulated Trend Toward Current Operating Condition")
trend_var = st.selectbox("Select variable for trend view", [VARIABLE_META[k]["label"] for k in VARIABLES], index=0)
trend_key = next(k for k in VARIABLES if VARIABLE_META[k]["label"] == trend_var)
plot_df = trend_df[["Minute", trend_key]].rename(columns={trend_key: VARIABLE_META[trend_key]["label"]})
st.line_chart(plot_df.set_index("Minute"))

st.subheader("How this demo works")
st.markdown(
    "This local application is a **demo-oriented surrogate** of a CNN-BiGRU-based early fault diagnosis system. "
    "It uses manually adjusted process variables, a profile-matching diagnosis engine, confidence-based early warning time estimation, "
    "and interactive visualizations to simulate how an AI model can identify faults earlier than conventional alarms."
)

st.caption("Run locally with: streamlit run fault_diagnosis_demo_app.py")

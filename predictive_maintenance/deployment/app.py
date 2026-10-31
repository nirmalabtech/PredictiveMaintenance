import json
import os

import joblib
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "best_model.joblib")
META_PATH = os.path.join(HERE, "model_metadata.json")

# Sensor inputs: column name -> (label, min, max, default, step, help). Limits mirror the cleaning rules.
SENSORS = {
    "engine_rpm": ("Engine speed (rpm)", 1.0, 5000.0, 746.0, 10.0, "Typical range in training data: 61 - 2,239 rpm"),
    "lub_oil_pressure": ("Lubricating-oil pressure (bar)", 0.01, 15.0, 3.16, 0.1, "Typical range: 0.00 - 7.27 bar"),
    "fuel_pressure": ("Fuel pressure (bar)", 0.01, 30.0, 6.20, 0.1, "Typical range: 0.00 - 21.14 bar"),
    "coolant_pressure": ("Coolant pressure (bar)", 0.01, 10.0, 2.17, 0.1, "Typical range: 0.00 - 7.48 bar"),
    "lub_oil_temp": ("Lubricating-oil temperature (°C)", 1.0, 150.0, 76.82, 0.5, "Typical range: 71.32 - 89.58 °C"),
    "coolant_temp": ("Coolant temperature (°C)", 1.0, 150.0, 78.35, 0.5, "Typical range: 61.67 - 102 °C"),
}
RAW_NAMES = {"Engine rpm": "engine_rpm", "Lub oil pressure": "lub_oil_pressure", "Fuel pressure": "fuel_pressure",
             "Coolant pressure": "coolant_pressure", "lub oil temp": "lub_oil_temp", "Coolant temp": "coolant_temp"}

st.set_page_config(page_title="Engine Predictive Maintenance", page_icon=":wrench:", layout="centered")
st.title("Engine Predictive Maintenance")
st.write("Enter the latest engine sensor readings to estimate the risk that the engine **needs maintenance**.")


@st.cache_resource
def load_model():
    """Load the model committed to the repository by the GitHub Actions pipeline."""
    if not os.path.exists(MODEL_PATH):
        st.error("Trained model not found. Run the GitHub Actions pipeline so it commits "
                 "`best_model.joblib` to `predictive_maintenance/deployment/`, then reload.")
        st.stop()
    meta = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {}
    return joblib.load(MODEL_PATH), meta


model, meta = load_model()
features = meta.get("features", list(SENSORS))

with st.sidebar:
    st.header("Deployed model")
    st.write(f"**Algorithm:** {meta.get('model_name', 'unknown')}")
    tm = meta.get("test_metrics", {})
    if tm:
        st.write(f"**Test recall:** {tm.get('test_recall', 0):.2f}  \n**Test precision:** {tm.get('test_precision', 0):.2f}"
                 f"  \n**Test ROC-AUC:** {tm.get('test_roc_auc', 0):.2f}")
    trained_by = meta.get("trained_by", "unknown")
    run_no = meta.get("github_run_number")
    st.write(f"**Trained by:** {trained_by}" + (f" (run #{run_no})" if run_no else ""))
    st.header("Alert setting")
    threshold = st.slider("Alert threshold", 0.30, 0.70, float(meta.get("decision_threshold", 0.5)), 0.05,
                          help="Lower = catch more faults but raise more inspections; higher = fewer inspections, more missed faults.")

with st.form("sensor_form"):
    st.subheader("Sensor readings")
    col1, col2 = st.columns(2)
    values = {}
    for i, (name, (label, lo, hi, default, step, help_text)) in enumerate(SENSORS.items()):
        with (col1 if i % 2 == 0 else col2):
            values[name] = st.number_input(label, min_value=lo, max_value=hi, value=default, step=step, help=help_text)
    submitted = st.form_submit_button("Predict engine condition")

if submitted:
    # Save the inputs into a single-row dataframe with exactly the training column names and order.
    input_df = pd.DataFrame([values])[features]
    st.subheader("Model input")
    st.dataframe(input_df, hide_index=True, width="stretch")

    probability = float(model.predict_proba(input_df)[0, 1])
    st.subheader("Prediction")
    st.metric("Maintenance-risk probability", f"{probability:.1%}")
    st.progress(min(max(probability, 0.0), 1.0))
    if probability >= threshold:
        st.error(f"**Needs maintenance** (risk {probability:.1%} ≥ alert threshold {threshold:.0%}). "
                 "Schedule an inspection - prioritise the fuel system and idle stability checks.")
    else:
        st.success(f"**Operating normally** (risk {probability:.1%} < alert threshold {threshold:.0%}). "
                   "Continue routine monitoring.")

with st.expander("Batch prediction for a fleet (CSV upload)"):
    st.write("Upload a CSV with the six sensor columns (original or snake_case names).")
    upload = st.file_uploader("Sensor readings CSV", type="csv")
    if upload is not None:
        batch = pd.read_csv(upload).rename(columns=RAW_NAMES)
        missing = [c for c in features if c not in batch.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            out = batch.copy()
            out["maintenance_risk"] = model.predict_proba(batch[features])[:, 1].round(4)
            out["alert"] = (out["maintenance_risk"] >= threshold).map({True: "Needs maintenance", False: "Normal"})
            st.write(f"{(out['alert'] == 'Needs maintenance').sum():,} of {len(out):,} engines flagged for maintenance.")
            st.dataframe(out.sort_values("maintenance_risk", ascending=False), hide_index=True, width="stretch")
            st.download_button("Download predictions", out.to_csv(index=False), "engine_predictions.csv", "text/csv")

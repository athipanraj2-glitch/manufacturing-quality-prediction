from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent

# ---------- Load model ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE / "models" / "logistic_regression.joblib")
    columns = joblib.load(BASE / "models" / "feature_columns.joblib")
    return model, columns

model, feature_columns = load_artifacts()

# ---------- Page ----------
st.set_page_config(page_title="CNC Defect Predictor", page_icon="🏭")
st.title("🏭 CNC Part Defect Predictor")
st.write("Enter the Machine sensor values. Model will predict defect risk.")
st.caption("⚠️ Model trained on simulated data (demo purpose).")

# ---------- Sidebar inputs ----------
st.sidebar.header("Machine Readings")

material = st.sidebar.selectbox("Material", ["Aluminium", "Steel", "Titanium"])
spindle_speed = st.sidebar.slider("Spindle speed (rpm)", 600, 2500, 1500)
feed_rate = st.sidebar.slider("Feed rate (mm/min)", 80, 350, 200)
cutting_temp = st.sidebar.slider("Cutting temperature (°C)", 100, 350, 200)
vibration = st.sidebar.slider("Vibration (mm/s)", 0.1, 6.0, 2.5, step=0.1)
tool_wear = st.sidebar.slider("Tool wear (min)", 0, 240, 100)
coolant = st.sidebar.slider("Coolant flow (L/min)", 2.0, 14.0, 8.0, step=0.5)

threshold = st.sidebar.slider("Decision threshold", 0.1, 0.9, 0.3, step=0.05)

# ---------- Feature engineering (training oda EXACT same) ----------
row = {
    "spindle_speed_rpm": spindle_speed,
    "feed_rate_mm_min": feed_rate,
    "cutting_temp_c": cutting_temp,
    "vibration_mm_s": vibration,
    "tool_wear_min": tool_wear,
    "coolant_flow_lpm": coolant,
    "cutting_intensity": spindle_speed * feed_rate / 1000,
    "wear_x_vibration": tool_wear * vibration,
    "heat_per_coolant": cutting_temp / coolant,
    "vibration_per_speed": vibration / spindle_speed * 1000,
    "material_type_Steel": int(material == "Steel"),
    "material_type_Titanium": int(material == "Titanium"),
}
input_df = pd.DataFrame([row])[feature_columns]   # column order match

# ---------- Predict ----------
prob = model.predict_proba(input_df)[0, 1]

st.subheader("Prediction")
col1, col2 = st.columns(2)
col1.metric("Defect risk", f"{prob * 100:.1f}%")
col2.metric("Threshold", f"{threshold:.2f}")

st.progress(min(float(prob), 1.0))

if prob >= threshold:
    st.error("🚨 HIGH RISK: Inspect the Part  / check the tool.")
else:
    st.success("✅ LOW RISK: Part OK.")

with st.expander("Features sent inside the model"):
    st.dataframe(input_df.T.rename(columns={0: "value"}))
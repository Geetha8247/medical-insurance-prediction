import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a5f;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1e3a5f, #2e6da4);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin-top: 1.5rem;
    }
    .prediction-label {
        font-size: 1rem;
        opacity: 0.85;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.4rem 0;
    }
    .prediction-note {
        font-size: 0.85rem;
        opacity: 0.7;
    }
    .stButton > button {
        width: 100%;
        background-color: #1e3a5f;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        margin-top: 1rem;
        transition: background 0.3s;
    }
    .stButton > button:hover {
        background-color: #2e6da4;
    }
    .info-card {
        background: #f0f7ff;
        border-left: 4px solid #2e6da4;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
        font-size: 0.9rem;
        color: #1e3a5f;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e3a5f;
        border-bottom: 2px solid #e0e9f4;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
config  = joblib.load("feature_config.pkl")
        return model, preprocessor, config
    except FileNotFoundError:
        return None, None, None


model, preprocessor, config = load_artifacts()


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏥 Medical Insurance Cost Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Estimate your annual insurance charges instantly</div>', unsafe_allow_html=True)

if model is None:
    st.error("❌ Model files not found! Please place `model.pkl`, `preprocessor.pkl`, "
             "and `feature_config.pkl` inside a `models/` folder.")
    st.info("Run the Colab notebook first to generate the model files, then download them.")
    st.stop()

st.markdown(f'<div class="info-card">🤖 Model: <b>{config["best_model_name"]}</b> — '
            f'trained on 1,338 insurance records</div>', unsafe_allow_html=True)


# ─── Input Form ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=65, value=30, step=1,
                    help="Age of the primary beneficiary")
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=55.0,
                           value=27.0, step=0.1, format="%.1f",
                           help="Weight(kg) / Height(m)²")
    children = st.selectbox("Number of Children / Dependents",
                             options=[0, 1, 2, 3, 4, 5], index=0)

with col2:
    sex = st.selectbox("Sex", options=["male", "female"])
    smoker = st.selectbox("Smoker?", options=["no", "yes"],
                           help="Smoking has the strongest impact on insurance cost")
    region = st.selectbox("US Region", options=["northeast", "northwest", "southeast", "southwest"])

# BMI guide
bmi_label = (
    "Underweight" if bmi < 18.5 else
    "Normal weight" if bmi < 25 else
    "Overweight" if bmi < 30 else "Obese"
)
bmi_color = {"Underweight": "🔵", "Normal weight": "🟢", "Overweight": "🟡", "Obese": "🔴"}
st.caption(f"BMI category: {bmi_color[bmi_label]} **{bmi_label}**")


# ─── Predict ─────────────────────────────────────────────────────────────────
predict_btn = st.button("🔮 Predict Insurance Cost")

if predict_btn:
    try:
        # Engineer features (must match notebook logic exactly)
        bmi_cat_map = {
            bmi < 18.5: "Underweight",
            18.5 <= bmi < 25: "Normal",
            25 <= bmi < 30: "Overweight",
            bmi >= 30: "Obese"
        }
        bmi_category = bmi_cat_map[True]

        age_group = (
            "Young" if age <= 25 else
            "Middle" if age <= 40 else
            "Senior" if age <= 55 else "Elderly"
        )

        family_size = (
            "No_children" if children == 0 else
            "Small_family" if children <= 2 else "Large_family"
        )

        smoker_obese = int(smoker == "yes" and bmi >= 30)
        age_bmi = age * bmi

        input_data = pd.DataFrame([{
            "age": age,
            "bmi": bmi,
            "children": children,
            "age_bmi": age_bmi,
            "smoker_obese": smoker_obese,
            "sex": sex,
            "smoker": smoker,
            "region": region,
            "bmi_category": bmi_category,
            "age_group": age_group,
            "family_size": family_size
        }])

        # Preprocess and predict
        input_processed = preprocessor.transform(input_data)
        prediction = model.predict(input_processed)[0]
        monthly = prediction / 12

        # Display result
        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-label">Estimated Annual Insurance Cost</div>
            <div class="prediction-value">${prediction:,.2f}</div>
            <div class="prediction-note">≈ ${monthly:,.2f} per month</div>
        </div>
        """, unsafe_allow_html=True)

        # Risk factors
        st.markdown("---")
        st.markdown("**📊 Key Risk Factors in Your Profile:**")

        factors = []
        if smoker == "yes":
            factors.append(("🚬 Smoker", "HIGH", "Smoking increases cost by ~4x on average"))
        if bmi >= 30:
            factors.append(("⚖️ Obese BMI", "MEDIUM", f"BMI {bmi:.1f} — Obese range"))
        if age >= 50:
            factors.append(("🎂 Age", "MEDIUM", f"Age {age} — Senior bracket"))
        if children >= 3:
            factors.append(("👨‍👩‍👧‍👦 Large family", "LOW", f"{children} dependents covered"))
        if not factors:
            factors.append(("✅ Low risk profile", "LOW", "No major risk factors detected"))

        for factor, level, note in factors:
            color = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#27ae60"}[level]
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;padding:8px;"
                f"border-radius:8px;border-left:4px solid {color};background:#f9f9f9;"
                f"margin-bottom:8px'>"
                f"<span style='font-weight:600'>{factor}</span>"
                f"<span style='background:{color};color:white;padding:2px 8px;"
                f"border-radius:12px;font-size:0.75rem'>{level}</span>"
                f"<span style='color:#666;font-size:0.85rem'>{note}</span></div>",
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.info("Please ensure all input fields are filled correctly.")


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("📌 This is an ML model prediction and not a real insurance quote. "
           "Actual costs vary by insurer.")

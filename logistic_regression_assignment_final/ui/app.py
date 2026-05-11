"""
Assignment 01 — Logistic Regression UI
Iris Flower Classifier — Streamlit App
Author: su92-bsaim-s23-015
"""

import streamlit as st
import numpy as np
import pickle
import os
from PIL import Image

# ── Page Config ──────────────────────────────
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# ── Load Model ────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), '../saved_model/logistic_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../saved_model/scaler.pkl')

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

CLASS_NAMES = ['Setosa', 'Versicolor', 'Virginica']
CLASS_EMOJI = ['🌼', '🌺', '🌸']
CLASS_INFO  = {
    'Setosa':     'Small petals, easily distinguishable. Found in Arctic and alpine regions.',
    'Versicolor': 'Medium-sized flower, common in eastern North America.',
    'Virginica':  'Largest petals among the three. Found in the eastern United States.',
}
CLASS_COLOR = {
    'Setosa':     '#E63946',
    'Versicolor': '#457B9D',
    'Virginica':  '#2A9D8F',
}

# ── Header ────────────────────────────────────
st.title("🌸 Iris Flower Classifier")
st.markdown("**Assignment 01 — Logistic Regression Classification**")
st.markdown("*Roll No: su92-bsaim-s23-015*")
st.markdown("---")

if not model_loaded:
    st.error("⚠️ Model not found! Please run `python model/train_model.py` first.")
    st.stop()

# ── Input Section ─────────────────────────────
st.subheader("📐 Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width  = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)

with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
    petal_width  = st.slider("Petal Width (cm)",  0.1, 2.5, 1.2, 0.1)

# ── Prediction ────────────────────────────────
st.markdown("---")
if st.button("🔍 Predict Species", use_container_width=True):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    name  = CLASS_NAMES[prediction]
    emoji = CLASS_EMOJI[prediction]
    color = CLASS_COLOR[name]
    info  = CLASS_INFO[name]

    st.markdown(
        f"""
        <div style="background:{color}22; border-left:5px solid {color};
                    padding:20px; border-radius:8px; margin-top:10px;">
            <h2 style="color:{color}; margin:0">{emoji} {name}</h2>
            <p style="margin:8px 0 0 0; color:#444">{info}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("📊 Prediction Probabilities")
    for i, (cls, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
        st.write(f"{CLASS_EMOJI[i]} **{cls}**")
        st.progress(float(prob), text=f"{prob * 100:.1f}%")

# ── Plots Section ─────────────────────────────
st.markdown("---")
st.subheader("📈 Model Evaluation")

CONF_PATH = os.path.join(os.path.dirname(__file__), '../model/plots/confusion_matrix.png')
FEAT_PATH = os.path.join(os.path.dirname(__file__), '../model/plots/feature_distributions.png')

tab1, tab2 = st.tabs(["Confusion Matrix", "Feature Distributions"])

with tab1:
    if os.path.exists(CONF_PATH):
        st.image(CONF_PATH, caption="Confusion Matrix on Test Set", use_column_width=True)
    else:
        st.info("Run train_model.py to generate plots.")

with tab2:
    if os.path.exists(FEAT_PATH):
        st.image(FEAT_PATH, caption="Feature Distributions by Species", use_column_width=True)
    else:
        st.info("Run train_model.py to generate plots.")

# ── Footer ────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888'>Assignment 01 · Logistic Regression · "
    "su92-bsaim-s23-015</p>",
    unsafe_allow_html=True
)

"""
app/app.py
Streamlit web app for DeepTruth — fake news detection.
"""

import os
import sys
import pickle
import numpy as np
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
APP_DIR  = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
SRC_DIR  = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from preprocess import clean_text, encode_texts
from model import load_trained_model

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepTruth — Misinformation Detector",
    page_icon="🔍",
    layout="centered",
)

# ── Load model & tokenizer (cached) ──────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_resources():
    model_path = os.path.join(ROOT_DIR, "models", "fake_news_model.h5")
    tok_path   = os.path.join(ROOT_DIR, "models", "tokenizer.pkl")

    if not os.path.exists(model_path):
        st.error("Model not found. Please run `python src/train.py` first.")
        st.stop()
    if not os.path.exists(tok_path):
        st.error("Tokenizer not found. Please run `python src/train.py` first.")
        st.stop()

    model = load_trained_model(model_path)

    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


def predict(text: str, model, tokenizer):
    """Return (label_str, confidence) for a raw text input."""
    X      = encode_texts([text], tokenizer)
    prob   = float(model.predict(X, verbose=0)[0][0])
    label  = "REAL ✅" if prob >= 0.5 else "FAKE ❌"
    conf   = prob if prob >= 0.5 else 1 - prob
    return label, conf, prob


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔍 DeepTruth")
st.subheader("LSTM-Based Misinformation Detector")
st.markdown("---")

model, tokenizer = load_resources()

# Input
# Initialize session state
# Initialize session state
# Initialize state
if "article_text" not in st.session_state:
    st.session_state.article_text = ""

# Create placeholder container
input_container = st.empty()

# Buttons
col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🔎 Predict", use_container_width=True)

with col2:
    clear_btn = st.button("🧹 Clear", use_container_width=True)

# Clear logic
if clear_btn:
    st.session_state.article_text = ""
    input_container.empty()   # 🔥 clears UI
    st.rerun()

# Render input INSIDE container
with input_container:
    article = st.text_area(
        "Paste a news article or headline below:",
        height=200,
        value=st.session_state.article_text
    )
# Buttons
col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🔎 Predict", use_container_width=True)

with col2:
    clear_btn = st.button("🧹 Clear", use_container_width=True)

# Clear logic
if clear_btn:
    st.session_state.article_text = ""
    st.rerun()

if predict_btn:
    if not article.strip():
        st.warning("Please enter some text before predicting.")
    else:
        with st.spinner("Analysing…"):
            label, confidence, raw_prob = predict(article, model, tokenizer)

        st.markdown("---")
        st.markdown("### Result")

        # Verdict
        if "REAL" in label:
            st.success(f"**{label}**")
        else:
            st.error(f"**{label}**")

        # Confidence bar
        st.markdown(f"**Confidence:** `{confidence:.2%}`")
        st.progress(float(confidence))

        # Raw probability
        with st.expander("ℹ️ Raw probabilities"):
            st.write(f"P(Real) = `{raw_prob:.4f}`")
            st.write(f"P(Fake) = `{1 - raw_prob:.4f}`")

        # Word count
        word_count = len(article.split())
        if word_count < 20:
            st.info("⚠️ Short text may reduce prediction accuracy. Try a longer article.")

st.markdown("---")
st.caption(
    "DeepTruth uses an LSTM trained on the ISOT Fake News Dataset. "
    "Results are probabilistic — always verify with authoritative sources."
)

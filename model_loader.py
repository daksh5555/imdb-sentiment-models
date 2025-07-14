# --- Caching for performance ---
import streamlit as st
import joblib
from model_architectures import build_model_small, build_model_large

@st.cache_resource
def load_model_a():
    model = build_model_small()
    model.load_weights("model_6_1.weights.h5")  # Adjust path if needed
    return model

@st.cache_resource
def load_model_b():
    model = build_model_large()
    model.load_weights("model_33_1.weights.h5")  # Adjust path if needed
    return model

@st.cache_resource
def load_tokenizer_a():
    return joblib.load("tokenizer_50k.pkl")

@st.cache_resource
def load_tokenizer_b():
    return joblib.load("tokenizer_255k.pkl")

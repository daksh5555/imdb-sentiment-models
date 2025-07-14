import streamlit as st
import tensorflow as tf
import json
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from model_architectures import build_model_small, build_model_large
from model_loader import load_model_a, load_model_b, load_tokenizer_a, load_tokenizer_b

import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials

import gspread
from oauth2client.service_account import ServiceAccountCredentials

def save_to_google_sheet(data):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]

    # Convert Streamlit's AttrDict to a normal dict (correct way)
    creds_dict = {k: v for k, v in st.secrets["gcp_credentials"].items()}

    # Handle multiline private key properly
    if "private_key" in creds_dict:
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")

    # Authenticate and connect
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open("Sentiment Feedback Log").sheet1

    # Append row
    sheet.append_row([
        data.get("timestamp", ""),
        data.get("username", ""),
        data.get("user_id", ""),
        data.get("text", ""),
        data.get("model_a", ""),
        data.get("model_b", ""),
        data.get("ensemble", ""),
        data.get("feedback", "")
    ])


st.set_page_config(page_title="Sentiment Model Comparison", layout="wide")
st.title("📊 Sentiment Classifier Comparison")

# --- Load models and tokenizers ---
model_a = load_model_a()  # 6.5M params
model_b = load_model_b()  # 34M params
tokenizer_a = load_tokenizer_a() #50k vocab
tokenizer_b = load_tokenizer_b() #256k vocab

# --- Constants ---
maxlen = 300
labels = ["Negative", "Neutral", "Positive"]

# --- Preprocess ---
def preprocess(text, tokenizer):
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen, padding='post')
    return padded

# --- Format Output ---
def format_probs(probs):
    return {labels[i]: f"{probs[i]*100:.2f}%" for i in range(3)}

# --- Text Input ---
st.markdown("### 📝 Enter a review:")
text = st.text_area("", height=150)

# --- File Upload ---
st.markdown("---")
file = st.file_uploader("📂 Or upload a CSV file with a 'review' column for bulk analysis", type=["csv"])

# Optional: User identification
user_name = st.text_input("🔐 Enter your name:")
user_id = st.text_input("🔐 Enter your email (optional):")

pred_a = pred_b = ensemble_label = None

if st.button("🔍 Analyze") and (text.strip() or file):
    if text.strip():
        padded_a = preprocess(text, tokenizer_a)
        padded_b = preprocess(text, tokenizer_b)
        pred_a = model_a.predict(padded_a)[0]
        pred_b = model_b.predict(padded_b)[0]
        ensemble_pred = (pred_a + pred_b) / 2
        ensemble_label = labels[int(ensemble_pred.argmax())]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🔹 Model A")
            st.caption("🧠 6M Parameters | 📖 50k Vocab")
            st.markdown(" | ".join([f"**{l}:** {v}" for l, v in format_probs(pred_a).items()]))
            st.write(f"→ **Predicted:** _{labels[int(pred_a.argmax())]}_")

        with col2:
            st.subheader("🔸 Model B")
            st.caption("🧠 34M Parameters | 📖 256k Vocab")
            st.markdown(" | ".join([f"**{l}:** {v}" for l, v in format_probs(pred_b).items()]))
            st.write(f"→ **Predicted:** _{labels[int(pred_b.argmax())]}_")

        with col3:
            st.subheader("⚖️ Ensemble Average")
            st.caption("🧮 Averaged Output (A + B)")
            st.markdown(" | ".join([f"**{l}:** {v}" for l, v in format_probs(ensemble_pred).items()]))
            st.write(f"→ **Final Sentiment:** ✅ _{ensemble_label}_")

        st.markdown("### 📈 Confidence Comparison")
        st.bar_chart({
            "Model A": pred_a,
            "Model B": pred_b,
            "Ensemble": ensemble_pred
        })

    if file:
        df = pd.read_csv(file)
        if 'review' not in df.columns:
            st.error("CSV must contain a 'review' column.")
        else:
            preds = []
            for text in df['review']:
                padded_a = preprocess(text, tokenizer_a)
                padded_b = preprocess(text, tokenizer_b)
                pred_a = model_a.predict(padded_a)[0]
                pred_b = model_b.predict(padded_b)[0]
                ensemble = (pred_a + pred_b) / 2
                preds.append(labels[int(ensemble.argmax())])

            df['Predicted Sentiment'] = preds
            st.dataframe(df)
            st.download_button("📥 Download Results", df.to_csv(index=False), file_name="sentiment_predictions.csv")

# --- Info Panel ---
with st.expander("ℹ️ Model Details"):
    st.markdown("""
    - **Model A**: Smaller model, faster, trained on 50k vocab.
    - **Model B**: Larger model, more accurate, trained on 256k vocab.
    - Ensemble averages predictions from both.
    """)

# --- Feedback ---
st.markdown("---")
st.markdown("### 💬 Feedback")
feedback = st.radio("Was the prediction helpful?", ["👍 Yes", "👎 No", "No comment"], horizontal=True)

if feedback and (user_name.strip() or user_id.strip() or text.strip()):
    st.success("Thanks for your feedback! ✅")

    feedback_data = {
        "timestamp": datetime.now().isoformat(),
        "username": user_name,
        "user_id": user_id,
        "text": text if text else None,
        "model_a": labels[int(pred_a.argmax())] if pred_a is not None else None,
        "model_b": labels[int(pred_b.argmax())] if pred_b is not None else None,
        "ensemble": ensemble_label if ensemble_label is not None else None,
        "feedback": feedback if feedback != "No comment" else None,
    }

    # Save to local CSV
    log_path = "user_feedback.csv"
    feedback_df = pd.DataFrame([feedback_data])
    if not os.path.exists(log_path):
        feedback_df.to_csv(log_path, index=False)
    else:
        feedback_df.to_csv(log_path, mode='a', header=False, index=False)

    # Save to Google Sheets
    try:
        save_to_google_sheet(feedback_data)
    except Exception as e:
        st.error(f"Error saving feedback to Google Sheets: {e}")


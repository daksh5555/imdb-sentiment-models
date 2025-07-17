# 🎬 imdb-sentiment-models

Two deep learning models trained on a custom-scraped IMDb movie reviews dataset for multi-class sentiment classification: **Negative**, **Neutral**, and **Positive**.

---

## 📚 IMDB Sentiment Classifier (Dual-Model)

This repository contains **two Keras deep learning models**, each trained with a different vocabulary size and number of parameters to compare performance trade-offs (speed vs accuracy).

---

## 📂 Dataset & Training Notes

- ~150,000 IMDB reviews scraped from multiple movies manually.
- Pseudo-labeled using soft probability outputs from `cardiffnlp/twitter-roberta-base-sentiment`.
- This technique provides **soft targets** (probabilistic labels), helping models learn nuanced sentiment distributions.

---

## 🧠 Models

### 🔹 Model A — Lightweight
- File: `sentiment_model_imdb_6.6M.keras`  
- **Trainable Params**: ~6.6M  
- **Total Params**: ~13M  
- **Vocab Size**: 50,000  
- ✅ Faster, compact, suitable for limited-resource environments.

### 🔸 Model B — Larger Capacity
- File: `sentiment_model_imdb_34M.keras`  
- **Trainable Params**: ~34M  
- **Total Params**: ~99M  
- **Vocab Size**: 256,000  
- ✅ More expressive, higher accuracy on complex reviews.

---

## 🔤 Tokenizers

Each model has its dedicated tokenizer:

| Model | Tokenizer |
|-------|-----------|
| A     | `tokenizer_50k.json` |
| B     | `tokenizer_256k.json` |

Both are saved in Keras JSON format.

---

## 🧪 Load Models & Tokenizers from 🤗 Hub

```python
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Load Model A
model_a_path = hf_hub_download("Daksh0505/sentiment-model-imdb", "sentiment_model_imdb_6.6M.keras")
tokenizer_a_path = hf_hub_download("Daksh0505/sentiment-model-imdb", "tokenizer_50k.json")
with open(tokenizer_a_path) as f:
    tokenizer_a = tokenizer_from_json(json.load(f))
model_a = load_model(model_a_path)

# Load Model B
model_b_path = hf_hub_download("Daksh0505/sentiment-model-imdb", "sentiment_model_imdb_34M.keras")
tokenizer_b_path = hf_hub_download("Daksh0505/sentiment-model-imdb", "tokenizer_256k.json")
with open(tokenizer_b_path) as f:
    tokenizer_b = tokenizer_from_json(json.load(f))
model_b = load_model(model_b_path)
```

## 🚀 Try the Live Demo

Click below to test both models live in your browser:

[![Open in Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Sentiment%20Demo-blue?logo=streamlit&style=for-the-badge)](https://huggingface.co/spaces/Daksh0505/sentiment-model-comparison)

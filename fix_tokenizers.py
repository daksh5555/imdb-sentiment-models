# fix_tokenizers_json.py
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
import joblib

# Load your existing tokenizers
tokenizer_a = joblib.load("tokenizer_50k.pkl")
tokenizer_b = joblib.load("tokenizer_255k.pkl")

# Save as JSON (portable)
with open("tokenizer_50k.json", "w") as f:
    f.write(tokenizer_a.to_json())

with open("tokenizer_255k.json", "w") as f:
    f.write(tokenizer_b.to_json())

print("✅ Saved tokenizers as JSON for portability.")

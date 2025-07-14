# fix_tokenizers.py
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer

# Load old tokenizer (may fail if incompatible)
old_tokenizer_50k = joblib.load("tokenizer_50k.pkl")
old_tokenizer_255k = joblib.load("tokenizer_255k.pkl")

# Re-save using correct Tokenizer class
joblib.dump(old_tokenizer_50k, "tokenizer_50k.pkl")
joblib.dump(old_tokenizer_255k, "tokenizer_255k.pkl")

print("✅ Tokenizers re-saved successfully.")

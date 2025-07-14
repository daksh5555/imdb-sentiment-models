# Sentiment-Analyzer-IMDB
A Streamlit web app that compares two deep learning sentiment classifiers trained on web-scraped IMDB reviews using different vocab sizes and model sizes, with optional user feedback logging.

# 📊 Sentiment Model Comparison App

An interactive **Streamlit web application** for sentiment analysis of **IMDB movie reviews**, featuring a side-by-side comparison of two deep learning models trained with different vocab sizes and capacities.

---

## 🔍 Features

- **Model A**: Lightweight (6M parameters), trained on a 50k vocabulary – fast and efficient.
- **Model B**: Larger (34M parameters), trained on a 256k vocabulary – more expressive and accurate.
- **Ensemble Prediction**: Final sentiment based on averaging both models’ outputs.
- **Single or Bulk Analysis**:
  - Paste a single review in the text box.
  - Or upload a CSV file with a `review` column for batch prediction.
- **Interactive Results**:
  - Visualize prediction probabilities.
  - Download sentiment results as CSV.
- **Feedback Logging** (Optional):
  - Users can submit their name/email, and satisfaction rating.
  - Feedback is stored both locally (`user_feedback.csv`) and in Google Sheets via API.

---

## ⚙️ Built With

- 🧠 TensorFlow / Keras
- 🗃 Pandas & NumPy
- 🌐 Streamlit
- 📊 gspread (for Google Sheets logging)

---

## 🚀 Try It Live

👉 [Click here to open the app on Streamlit Cloud](#)  
*(Replace with your deployed app link)*



# Trendify Global — Multilingual Sentiment Analysis Dashboard

A Streamlit web app for analysing and predicting customer review sentiment across multiple languages using machine learning and transformer models.

## Live Demo
🚀 Deployed on [Streamlit Cloud] https://trendify-sentiment-zcezfpsutw6vstznbfmtvd.streamlit.app

---

## Overview
This project classifies customer reviews into **positive**, **neutral**, and **negative** sentiments. The dataset contains **30,000 reviews in multiple languages** from customers across different countries and product categories. A multilingual DistilBERT model was chosen to handle the language diversity in the data.

---

## Models
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Naive Bayes (MultinomialNB) | 100% | 1.0 |
| Logistic Regression | — | — |
| DistilBERT (multilingual) | 100% | 1.0 |

> ⚠️ Note: Perfect scores may indicate data leakage — currently under investigation.

---

## Features
- 📊 **Dashboard** — sentiment distribution, ratings, product categories, and country breakdowns
- 🔍 **Predict** — live sentiment prediction with confidence scores
- 🌍 **Multilingual support** — handles reviews in multiple languages via `distilbert-base-multilingual-cased`

---

## Dataset
| Property | Detail |
|----------|--------|
| Total samples | 30,000 |
| Train / Test split | 24,000 / 6,000 |
| Languages | Multiple |
| Classes | Positive, Neutral, Negative |
| Features | Review text, rating, product category, country |

---

## Project Structure
```
Trendify_Sentiment/
├── app.py
├── requirements.txt
├── .gitattributes              # Git LFS tracking config
├── data/
│   └── raw/
│       └── raw_reviews.csv
├── artifacts/
│   ├── models/
│   │   ├── naive_bayes_model.pkl
│   │   ├── logistic_regression_model.pkl
│   │   └── distilbert_model/       # stored via Git LFS (516MB)
│   ├── encoders/
│   │   └── target_encoder.pkl
│   └── tokenizers/
│       ├── tfidf_vectorizer.pkl
│       └── distilbert_tokenizer/
```

---

## Setup

```bash
# 1. Clone the repo (Git LFS required for large model files)
git lfs install
git clone https://github.com/msquare190/trendify-sentiment.git
cd trendify-sentiment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

> ⚠️ Make sure **Git LFS** is installed before cloning, otherwise the DistilBERT model files will not download correctly.
> Install it from: https://git-lfs.com

---

## Deployment Notes
- App hosted on **Streamlit Cloud**
- Large model files (516MB) managed via **Git LFS**
- Streamlit Cloud supports Git LFS automatically

---

## Tech Stack
`Python` `Streamlit` `PyTorch` `HuggingFace Transformers` `Scikit-learn` `NLTK` `Plotly` `Pandas` `Git LFS`

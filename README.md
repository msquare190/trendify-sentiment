# Trendify Global — Multilingual Sentiment Analysis Dashboard

A Streamlit web app for analysing and predicting customer review sentiment across multiple languages using machine learning and transformer models.

## Live Demo
🚀 Deployed on [Streamlit Cloud](https://streamlit.io/cloud)

---

## Overview
This project classifies customer reviews into **positive**, **neutral**, and **negative** sentiments. The dataset contains **45,000 reviews across 10 languages, 8 product categories, and 14 countries**. A multilingual DistilBERT model was chosen to handle the language diversity in the data.

---

## Features
- 📊 **Dashboard** — visualises sentiment distribution, ratings, product categories, and country breakdowns
- 🔍 **Predict** — live sentiment prediction with confidence scores across three models
- 🌍 **Multilingual support** — handles reviews in 10 languages via `distilbert-base-multilingual-cased`

---

## Dataset
| Property | Detail |
|----------|--------|
| Total rows | 45,000 |
| Unique raw reviews | 2,611 |
| Unique processed reviews | 155 |
| Train / Test split | 80% / 20% |
| Languages | 10 (EN, FR, ES, DE, and more) |
| Product categories | 8 |
| Countries | 14 |
| Classes | Positive, Neutral, Negative (balanced — 15,000 each) |
| Features | Review text, rating, product category, country |

> ⚠️ **Dataset Limitation:** After preprocessing (lowercasing, stopword removal, lemmatization), the dataset reduces to 155 unique text patterns. This suggests the data was synthetically generated or heavily templated, resulting in high model accuracy that reflects memorisation rather than generalisation. Future work includes sourcing a more diverse multilingual dataset for robust evaluation.

---

## Models
| Model | Accuracy | F1 Score (Macro) |
|-------|----------|-----------------|
| Naive Bayes (MultinomialNB) | 100% | 1.0 |
| Logistic Regression | 100% | 1.0 |
| DistilBERT (multilingual) | 100% | 1.0 |

> **Note:** Perfect scores are a direct result of the dataset limitation described above — not a reflection of true generalisation ability.

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

## Known Limitations & Future Work
- Dataset contains limited unique text patterns after preprocessing — future work includes sourcing a more diverse real-world multilingual dataset
- DistilBERT was trained on CPU which increases training time significantly
- Model evaluation metrics should be interpreted with caution given the dataset characteristics

---

## Tech Stack
`Python` `Streamlit` `PyTorch` `HuggingFace Transformers` `Scikit-learn` `NLTK` `Plotly` `Pandas` `Git LFS`

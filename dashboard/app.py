# app.py
# Streamlit dashboard + prediction app for Trendify Global sentiment analysis
# - Loads data from:      /data/raw/raw_reviews.csv
# - Loads artifacts from: /artifacts/...
# - Provides 2 pages: Dashboard + Predict

import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from plotly.subplots import make_subplots
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Optional (used in preprocessing pipeline to match your notebooks)
import nltk

warnings.filterwarnings("ignore")



# -----------------------------
# Paths (assumes app.py is inside /dashboard)
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "raw_reviews.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
ENCODERS_DIR = ARTIFACTS_DIR / "encoders"
TOKENIZERS_DIR = ARTIFACTS_DIR / "tokenizers"

NB_MODEL_PATH = MODELS_DIR / "naive_bayes_model.pkl"
LR_MODEL_PATH = MODELS_DIR / "logistic_regression_model.pkl"
VECTORIZER_PATH = TOKENIZERS_DIR / "tfidf_vectorizer.pkl"
TARGET_ENCODER_PATH = ENCODERS_DIR / "target_encoder.pkl"

DISTILBERT_MODEL_DIR = MODELS_DIR / "distilbert_model"
DISTILBERT_TOKENIZER_DIR = TOKENIZERS_DIR / "distilbert_tokenizer"


# -----------------------------
# Constants
# -----------------------------
SENTIMENT_ORDER = ["positive", "neutral", "negative"]
SENTIMENT_COLORS = {"positive": "green", "neutral": "gray", "negative": "red"}


# -----------------------------
# NLTK downloads (quiet)
# -----------------------------
@st.cache_resource
def ensure_nltk():
    try:
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    except Exception:
        pass
    return True


# -----------------------------
# Data + Artifacts loaders
# -----------------------------
@st.cache_data
def load_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "review" in df.columns:
        df["review"] = df["review"].astype(str)
        df["review_length"] = df["review"].str.len()
        df["word_count"] = df["review"].apply(lambda x: len(str(x).split()))

    return df


@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    artifacts = {
        "vectorizer": None,
        "label_encoder": None,
        "sklearn_models": {},
        "distilbert_model": None,
        "distilbert_tokenizer": None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    if VECTORIZER_PATH.exists():
        artifacts["vectorizer"] = joblib.load(VECTORIZER_PATH)

    if TARGET_ENCODER_PATH.exists():
        artifacts["label_encoder"] = joblib.load(TARGET_ENCODER_PATH)

    if NB_MODEL_PATH.exists():
        artifacts["sklearn_models"]["Naive Bayes (MultinomialNB)"] = joblib.load(NB_MODEL_PATH)

    if LR_MODEL_PATH.exists():
        artifacts["sklearn_models"]["Logistic Regression"] = joblib.load(LR_MODEL_PATH)

    try:
        distilbert_model = DistilBertForSequenceClassification.from_pretrained(
            "msquare190/trendify-distilbert"
        ).to(artifacts["device"])
        distilbert_model.eval()

        distilbert_tokenizer = DistilBertTokenizer.from_pretrained(
            "msquare190/trendify-distilbert"
        )

        artifacts["distilbert_model"] = distilbert_model
        artifacts["distilbert_tokenizer"] = distilbert_tokenizer
    except Exception:
        pass

    return artifacts


# -----------------------------
# Preprocessing
# -----------------------------
def clean_text(text: str) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stopwords(text: str) -> str:
    if not text or not text.strip():
        return ""

    try:
        sw = set(stopwords.words("english"))
    except Exception:
        sw = set()

    words = text.split()
    words = [w for w in words if w not in sw]
    return " ".join(words)


def lemmatize_text(text: str) -> str:
    if not text or not text.strip():
        return ""

    try:
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        words = [lemmatizer.lemmatize(w) for w in words]
        return " ".join(words)
    except Exception:
        return text


def preprocess_pipeline(text: str) -> str:
    cleaned = clean_text(text)
    no_sw = remove_stopwords(cleaned)
    lemm = lemmatize_text(no_sw)
    return lemm


# -----------------------------
# Helpers for safe label decoding
# -----------------------------
def normalize_label(label) -> str:
    return str(label).strip().lower()


def decode_single_prediction(pred, label_encoder=None):
    """
    Safely decode one sklearn model prediction.
    Handles:
    - already-string labels
    - numpy scalar ints
    - encoded labels that need inverse_transform
    """
    if isinstance(pred, str):
        return normalize_label(pred)

    if hasattr(pred, "item"):
        pred = pred.item()

    if label_encoder is not None:
        try:
            return normalize_label(label_encoder.inverse_transform([pred])[0])
        except Exception:
            pass

    return normalize_label(pred)


def decode_class_labels(class_values, label_encoder=None):
    """
    Safely decode class labels for sklearn probability outputs.
    Uses model.classes_ and only inverse-transforms if needed.
    """
    decoded = []
    for c in class_values:
        if isinstance(c, str):
            decoded.append(normalize_label(c))
        else:
            try:
                val = c.item() if hasattr(c, "item") else c
                if label_encoder is not None:
                    decoded.append(normalize_label(label_encoder.inverse_transform([val])[0]))
                else:
                    decoded.append(normalize_label(val))
            except Exception:
                decoded.append(normalize_label(c))
    return decoded


def get_distilbert_labels(model, num_classes: int):
    """
    Use DistilBERT's own config mapping to avoid label mix-ups with sklearn label encoders.
    """
    id2label = getattr(model.config, "id2label", None)

    if isinstance(id2label, dict) and len(id2label) == num_classes:
        labels = []
        for i in range(num_classes):
            label = id2label.get(i, id2label.get(str(i), str(i)))
            labels.append(normalize_label(label))
        return labels

    # Fallback to hardcoded labels
    return ["negative", "neutral", "positive"]


def order_probability_df(prob_df: pd.DataFrame) -> pd.DataFrame:
    if prob_df is None or prob_df.empty:
        return prob_df

    df = prob_df.copy()
    df["sentiment"] = df["sentiment"].astype(str).str.lower()

    ordered_labels = [s for s in SENTIMENT_ORDER if s in df["sentiment"].values]
    remaining_labels = [s for s in df["sentiment"].values if s not in ordered_labels]
    final_order = ordered_labels + [x for x in remaining_labels if x not in ordered_labels]

    df["_order"] = df["sentiment"].apply(
        lambda x: final_order.index(x) if x in final_order else len(final_order)
    )
    df = df.sort_values(["_order", "probability"], ascending=[True, False]).drop(columns="_order")
    df = df.reset_index(drop=True)
    return df


def sort_probability_df_desc(prob_df: pd.DataFrame) -> pd.DataFrame:
    if prob_df is None or prob_df.empty:
        return prob_df
    return prob_df.sort_values("probability", ascending=False).reset_index(drop=True)


def make_confidence_chart_df(prob_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep chart order stable across predictions.
    """
    return order_probability_df(prob_df)


# -----------------------------
# Plot helpers
# -----------------------------
def reorder_sentiment_index(index_vals):
    index_vals = [normalize_label(x) for x in index_vals]
    ordered = [s for s in SENTIMENT_ORDER if s in index_vals]
    remaining = [s for s in index_vals if s not in ordered]
    return ordered + remaining


def sentiment_distribution_fig(df: pd.DataFrame):
    sentiment_counts = df["sentiment"].astype(str).str.lower().value_counts()
    sentiment_counts = sentiment_counts.reindex(reorder_sentiment_index(sentiment_counts.index))

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Sentiment Distribution", "Sentiment Percentage"),
        specs=[[{"type": "bar"}, {"type": "pie"}]],
    )

    fig.add_trace(
        go.Bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            marker_color=[SENTIMENT_COLORS.get(s, "blue") for s in sentiment_counts.index],
            text=sentiment_counts.values,
            textposition="auto",
            name="Count",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            marker_colors=[SENTIMENT_COLORS.get(s, "blue") for s in sentiment_counts.index],
            textinfo="percent+label",
            hole=0.3,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(title_text="Sentiment Distribution Analysis", height=420, showlegend=False)
    return fig


def rating_distribution_fig(df: pd.DataFrame):
    rating_counts = df["rating"].value_counts().sort_index()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=rating_counts.index,
            y=rating_counts.values,
            text=rating_counts.values,
            textposition="auto",
        )
    )
    fig.update_layout(
        title_text="Rating Distribution",
        xaxis_title="Rating (Stars)",
        yaxis_title="Number of Reviews",
        height=420,
    )
    return fig


def rating_vs_sentiment_fig(df: pd.DataFrame):
    work_df = df.copy()
    work_df["sentiment"] = work_df["sentiment"].astype(str).str.lower()

    rating_sentiment = pd.crosstab(work_df["rating"], work_df["sentiment"])
    rating_sentiment_pct = pd.crosstab(work_df["rating"], work_df["sentiment"], normalize="index") * 100

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Rating vs Sentiment (Count)", "Rating vs Sentiment (Percentage)"),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )

    for sentiment in SENTIMENT_ORDER:
        if sentiment in rating_sentiment.columns:
            fig.add_trace(
                go.Bar(
                    name=sentiment.capitalize(),
                    x=rating_sentiment.index,
                    y=rating_sentiment[sentiment].values,
                    marker_color=SENTIMENT_COLORS[sentiment],
                ),
                row=1,
                col=1,
            )

    for sentiment in SENTIMENT_ORDER:
        if sentiment in rating_sentiment_pct.columns:
            fig.add_trace(
                go.Bar(
                    name=sentiment.capitalize(),
                    x=rating_sentiment_pct.index,
                    y=rating_sentiment_pct[sentiment].values,
                    marker_color=SENTIMENT_COLORS[sentiment],
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

    fig.update_layout(title_text="Rating and Sentiment Relationship", height=420, barmode="stack")
    return fig


def category_sentiment_fig(df: pd.DataFrame):
    work_df = df.copy()
    work_df["sentiment"] = work_df["sentiment"].astype(str).str.lower()

    category_sentiment = pd.crosstab(
        work_df["product_category"], work_df["sentiment"], normalize="index"
    ) * 100

    if "positive" in category_sentiment.columns:
        category_sentiment = category_sentiment.sort_values("positive", ascending=False)
    else:
        category_sentiment = category_sentiment.sort_index()

    fig = go.Figure()
    for sentiment in SENTIMENT_ORDER:
        if sentiment in category_sentiment.columns:
            fig.add_trace(
                go.Bar(
                    name=sentiment.capitalize(),
                    y=category_sentiment.index,
                    x=category_sentiment[sentiment].values,
                    orientation="h",
                    marker_color=SENTIMENT_COLORS[sentiment],
                    text=category_sentiment[sentiment].round(1).astype(str) + "%",
                    textposition="inside",
                )
            )

    fig.update_layout(
        title_text="Sentiment Distribution by Product Category",
        xaxis_title="Percentage (%)",
        yaxis_title="Product Category",
        barmode="stack",
        height=650,
        legend_title="Sentiment",
    )
    return fig


def top_countries_fig(df: pd.DataFrame, top_n: int = 15):
    country_counts = df["country"].value_counts().head(top_n)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=country_counts.index,
            y=country_counts.values,
            text=country_counts.values,
            textposition="auto",
        )
    )
    fig.update_layout(
        title_text=f"Top {top_n} Countries by Review Volume",
        xaxis_title="Country",
        yaxis_title="Number of Reviews",
        height=420,
    )
    return fig


def sentiment_by_country_fig(df: pd.DataFrame, top_n: int = 10):
    work_df = df.copy()
    work_df["sentiment"] = work_df["sentiment"].astype(str).str.lower()

    country_sentiment = pd.crosstab(
        work_df["country"], work_df["sentiment"], normalize="index"
    ) * 100

    if "positive" in country_sentiment.columns:
        country_sentiment = country_sentiment.sort_values("positive", ascending=False)
    else:
        country_sentiment = country_sentiment.sort_index()

    fig = go.Figure()
    top_index = country_sentiment.index[:top_n]

    for sentiment in SENTIMENT_ORDER:
        if sentiment in country_sentiment.columns:
            series = pd.to_numeric(country_sentiment.loc[top_index, sentiment], errors="coerce").fillna(0.0)
            vals = series.to_numpy(dtype=float)
            labels = [f"{v:.1f}%" for v in vals]

            fig.add_trace(
                go.Bar(
                    name=sentiment.capitalize(),
                    x=top_index,
                    y=vals,
                    marker_color=SENTIMENT_COLORS[sentiment],
                    text=labels,
                    textposition="inside",
                )
            )

    fig.update_layout(
        title_text=f"Sentiment Distribution by Country (Top {top_n})",
        xaxis_title="Country",
        yaxis_title="Percentage (%)",
        barmode="stack",
        height=420,
        legend_title="Sentiment",
    )
    return fig


# -----------------------------
# Prediction functions
# -----------------------------
def predict_sklearn(text, model, vectorizer, label_encoder=None):
    processed = preprocess_pipeline(text)
    X_vec = vectorizer.transform([processed])

    raw_pred = model.predict(X_vec)[0]
    pred_label = decode_single_prediction(raw_pred, label_encoder=label_encoder)

    prob_df = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_vec)[0]

        if hasattr(model, "classes_"):
            class_labels = decode_class_labels(model.classes_, label_encoder=label_encoder)
        else:
            class_labels = [str(i) for i in range(len(probs))]

        prob_df = pd.DataFrame(
            {
                "sentiment": class_labels,
                "probability": probs,
            }
        )

    return pred_label, prob_df, processed


def predict_distilbert(text, model, tokenizer, device="cpu"):
    """
    DistilBERT must use its own config.id2label mapping.
    Do not decode with the sklearn label encoder, or you risk label mix-ups.
    """
    inputs = tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

    pred_idx = int(torch.argmax(probabilities, dim=1).cpu().item())
    probs = probabilities.cpu().numpy()[0]

    class_labels = get_distilbert_labels(model, len(probs))
    pred_label = class_labels[pred_idx] if pred_idx < len(class_labels) else str(pred_idx)

    prob_df = pd.DataFrame(
        {
            "sentiment": class_labels,
            "probability": probs,
        }
    )

    return pred_label, prob_df


# -----------------------------
# Pages
# -----------------------------
def page_dashboard(df: pd.DataFrame):
    st.title("Trendify Global — Sentiment Dashboard")
    st.caption("Data source: /data/raw/raw_reviews.csv")

    with st.sidebar:
        st.subheader("Filters")

        categories = ["All"]
        if "product_category" in df.columns:
            categories += sorted([c for c in df["product_category"].dropna().unique()])

        countries = ["All"]
        if "country" in df.columns:
            countries += sorted([c for c in df["country"].dropna().unique()])

        sel_cat = st.selectbox("Product Category", categories, index=0)
        sel_country = st.selectbox("Country", countries, index=0)

    dff = df.copy()
    if "sentiment" in dff.columns:
        dff["sentiment"] = dff["sentiment"].astype(str).str.lower()

    if sel_cat != "All" and "product_category" in dff.columns:
        dff = dff[dff["product_category"] == sel_cat]
    if sel_country != "All" and "country" in dff.columns:
        dff = dff[dff["country"] == sel_country]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{len(dff):,}")

    if "sentiment" in dff.columns:
        c2.metric("Positive", f"{(dff['sentiment'] == 'positive').sum():,}")
        c3.metric("Neutral", f"{(dff['sentiment'] == 'neutral').sum():,}")
        c4.metric("Negative", f"{(dff['sentiment'] == 'negative').sum():,}")

    st.divider()

    if "sentiment" in dff.columns:
        st.plotly_chart(sentiment_distribution_fig(dff), use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        if "rating" in dff.columns:
            st.plotly_chart(rating_distribution_fig(dff), use_container_width=True)
    with colB:
        if "rating" in dff.columns and "sentiment" in dff.columns:
            st.plotly_chart(rating_vs_sentiment_fig(dff), use_container_width=True)

    if "product_category" in dff.columns and "sentiment" in dff.columns:
        st.plotly_chart(category_sentiment_fig(dff), use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        if "country" in dff.columns:
            st.plotly_chart(top_countries_fig(dff, top_n=15), use_container_width=True)
    with colD:
        if "country" in dff.columns and "sentiment" in dff.columns:
            st.plotly_chart(sentiment_by_country_fig(dff, top_n=10), use_container_width=True)

    with st.expander("Show data preview"):
        st.dataframe(dff.head(50), use_container_width=True)


def page_predict(artifacts):
    st.title("Sentiment Prediction")
    st.caption("Type a review, select a model, and get a sentiment prediction.")

    vectorizer = artifacts["vectorizer"]
    label_encoder = artifacts["label_encoder"]
    sklearn_models = artifacts["sklearn_models"]
    distilbert_model = artifacts["distilbert_model"]
    distilbert_tokenizer = artifacts["distilbert_tokenizer"]
    device = artifacts["device"]

    top_col1, top_col2 = st.columns([1, 1])
    with top_col1:
        if st.button("Reload artifacts / clear cache", use_container_width=True):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

    model_options = list(sklearn_models.keys())
    if distilbert_model is not None and distilbert_tokenizer is not None:
        model_options.append("DistilBERT")

    if not model_options:
        st.error("No models found in /artifacts/models/. Please ensure your model files exist.")
        return

    ensure_nltk()

    model_name = st.selectbox("Choose a model", model_options)

    text = st.text_area(
        "Customer review text",
        height=140,
        placeholder="e.g., Delivery was fast but the product quality was disappointing...",
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        predict_btn = st.button("Predict", type="primary", use_container_width=True)
    with col2:
        show_processed = st.checkbox("Show processed text", value=False)
    with col3:
        show_debug = st.checkbox("Show debug info", value=False)

    if predict_btn:
        if not text or not text.strip():
            st.warning("Please enter some text.")
            return

        st.subheader("Prediction")

        try:
            processed = None

            if model_name == "DistilBERT":
                pred_label, prob_df = predict_distilbert(
                    text=text,
                    model=distilbert_model,
                    tokenizer=distilbert_tokenizer,
                    device=device,
                )

                if show_processed:
                    st.code("(DistilBERT uses raw text input)", language="text")

            else:
                if vectorizer is None:
                    st.error("TF-IDF vectorizer not found.")
                    return

                model = sklearn_models[model_name]

                pred_label, prob_df, processed = predict_sklearn(
                    text=text,
                    model=model,
                    vectorizer=vectorizer,
                    label_encoder=label_encoder,
                )

                if show_processed:
                    st.code(processed or "(empty after preprocessing)", language="text")

            pred_label_lower = normalize_label(pred_label)

            if pred_label_lower == "positive":
                st.success(f"**{pred_label_lower.upper()}**")
            elif pred_label_lower == "negative":
                st.error(f"**{pred_label_lower.upper()}**")
            else:
                st.info(f"**{pred_label_lower.upper()}**")

            if prob_df is not None:
                chart_df = make_confidence_chart_df(prob_df)
                display_df = sort_probability_df_desc(prob_df.copy())

                st.subheader("Confidence")
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=chart_df["sentiment"],
                        y=chart_df["probability"],
                        marker_color=[SENTIMENT_COLORS.get(s, "blue") for s in chart_df["sentiment"]],
                        text=(chart_df["probability"] * 100).round(1).astype(str) + "%",
                        textposition="auto",
                    )
                )
                fig.update_layout(
                    yaxis_title="Probability",
                    xaxis_title="Sentiment",
                    yaxis=dict(range=[0, 1]),
                    height=360,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(display_df, use_container_width=True)

            if show_debug:
                with st.expander("Debug info", expanded=True):
                    st.write("Selected model:", model_name)

                    if model_name != "DistilBERT":
                        model = sklearn_models[model_name]
                        st.write("sklearn model.classes_:", getattr(model, "classes_", None))
                        if label_encoder is not None:
                            st.write(
                                "label_encoder classes_:",
                                getattr(label_encoder, "classes_", None),
                            )
                    else:
                        st.write("DistilBERT device:", device)
                        st.write("DistilBERT id2label:", getattr(distilbert_model.config, "id2label", None))
                        st.write("DistilBERT label2id:", getattr(distilbert_model.config, "label2id", None))

                    st.write("Predicted label:", pred_label_lower)

                    if prob_df is not None:
                        st.write("Raw probability table:")
                        st.dataframe(prob_df, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed for model: {model_name}")
            st.exception(e)


# -----------------------------
# Main App
# -----------------------------
def main():
    st.set_page_config(
        page_title="Trendify Global — Sentiment Dashboard",
        page_icon="🛒",
        layout="wide",
    )

    st.sidebar.title("Trendify Global")
    page = st.sidebar.radio("Navigate", ["Dashboard", "Predict"], index=0)

    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        st.stop()

    df = load_raw_data(DATA_PATH)

    required_cols = {"sentiment", "rating", "product_category", "country", "review"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"Some expected columns are missing in raw_reviews.csv: {missing}")

    if page == "Dashboard":
        if "sentiment" not in df.columns:
            st.error("Dashboard needs a 'sentiment' column in raw_reviews.csv.")
            st.stop()
        page_dashboard(df)

    elif page == "Predict":
        try:
            artifacts = load_artifacts()
        except Exception as e:
            st.error("Failed to load artifacts from /artifacts/.")
            st.exception(e)
            st.stop()

        page_predict(artifacts)


if __name__ == "__main__":
    main()




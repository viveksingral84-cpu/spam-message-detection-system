"""
Spam Message Detection System - Model Training & Evaluation
============================================================
This script trains a Machine Learning model using:
  1. TF-IDF (Term Frequency - Inverse Document Frequency) Vectorizer
  2. Multinomial Naive Bayes (MultinomialNB) Classifier

Dataset: UCI SMS Spam Collection
- Total Messages: 5,572
- Classes: 'ham' (Not Spam / Legitimate) and 'spam' (Spam)

Why Multinomial Naive Bayes?
----------------------------
Multinomial Naive Bayes is the gold-standard baseline algorithm for text
classification. It operates on Bayes' Theorem and calculates the probability of
a message belonging to a class based on the frequencies of the words it contains:
    P(Spam | Message) = P(Spam) * ∏ P(Word_i | Spam) / P(Message)
"""

import os
import json
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CSV_PATH = os.path.join(DATA_DIR, "spam.csv")
TSV_PATH = os.path.join(DATA_DIR, "sms.tsv")
MODEL_PATH = os.path.join(MODELS_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")

# Dataset download fallback URL (Public mirror of the UCI SMS Spam Collection)
DATASET_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"


def ensure_directories():
    """Ensure data/ and models/ directories exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def load_or_download_dataset():
    """
    Loads spam.csv if present.
    If not, downloads the UCI SMS Spam dataset TSV file, cleans it, and saves spam.csv.
    """
    ensure_directories()

    if os.path.exists(CSV_PATH):
        print(f"[*] Found existing dataset at: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        return df

    if os.path.exists(TSV_PATH):
        print(f"[*] Found existing TSV at: {TSV_PATH}. Converting to CSV...")
        df = pd.read_csv(TSV_PATH, sep="\t", header=None, names=["label", "message"])
        df.to_csv(CSV_PATH, index=False)
        return df

    print(f"[*] Dataset not found locally. Downloading from reliable public source:")
    print(f"    {DATASET_URL}")
    try:
        urllib.request.urlretrieve(DATASET_URL, TSV_PATH)
        df = pd.read_csv(TSV_PATH, sep="\t", header=None, names=["label", "message"])
        df.to_csv(CSV_PATH, index=False)
        print(f"[+] Downloaded and created: {CSV_PATH}")
        return df
    except Exception as e:
        raise RuntimeError(
            f"Failed to automatically download the dataset ({e}).\n"
            f"Please download the UCI SMS Spam Collection manually and place it as '{CSV_PATH}'\n"
            f"with columns: 'label' (ham/spam) and 'message' (SMS text)."
        )


def preprocess_data(df):
    """
    Cleans and prepares data:
    - Removes missing/null values
    - Trims whitespace
    - Maps labels: 'ham' -> 0 (Not Spam), 'spam' -> 1 (Spam)
    """
    print("\n--- 1. Data Exploration & Preprocessing ---")
    print(f"Initial row count: {len(df)}")

    # Ensure required columns exist
    if "label" not in df.columns or "message" not in df.columns:
        # If columns are named differently (e.g. v1, v2 from Kaggle format)
        if len(df.columns) >= 2:
            df = df.iloc[:, :2]
            df.columns = ["label", "message"]
        else:
            raise ValueError("Dataset must contain at least 'label' and 'message' columns.")

    # Drop null values
    df = df.dropna(subset=["label", "message"]).copy()
    df["message"] = df["message"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Filter only 'ham' and 'spam'
    df = df[df["label"].isin(["ham", "spam"])].copy()

    total = len(df)
    ham_count = (df["label"] == "ham").sum()
    spam_count = (df["label"] == "spam").sum()

    print(f"Cleaned row count: {total}")
    print(f" - Legitimate (Ham) messages: {ham_count} ({ham_count / total * 100:.2f}%)")
    print(f" - Spam messages:             {spam_count} ({spam_count / total * 100:.2f}%)")

    # Map labels to binary values (0: Ham, 1: Spam)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    return df


def train_and_evaluate(df):
    """
    Splits data, extracts TF-IDF features, trains MultinomialNB,
    evaluates performance, and saves artifacts.
    """
    X = df["message"]
    y = df["label_num"]

    # --- 2. Train / Test Split (Preventing Data Leakage) ---
    print("\n--- 2. Dataset Splitting ---")
    # 80% train, 20% test. Use stratify=y to maintain class distribution in both splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size:  {len(X_test)} samples")

    # --- 3. TF-IDF Feature Extraction ---
    print("\n--- 3. TF-IDF Feature Extraction ---")
    print("Fitting TF-IDF Vectorizer ONLY on training data (strictly preventing data leakage)...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),  # unigrams and bigrams (e.g. 'free' and 'free gift')
        max_features=5000,
        sublinear_tf=True     # replaces tf with 1 + log(tf) to dampen effect of repeated words
    )

    # CRITICAL: Fit and transform ONLY on training set
    X_train_tfidf = vectorizer.fit_transform(X_train)
    # Transform test set using the learned vocabulary from training set
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())} unique word/phrase features")

    # --- 4. Model Training: Multinomial Naive Bayes ---
    print("\n--- 4. Model Training ---")
    print("Training Multinomial Naive Bayes classifier (alpha=0.1 for Laplace smoothing)...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)
    print("[+] Model training completed successfully.")

    # --- 5. Model Evaluation ---
    print("\n--- 5. Model Evaluation on Unseen Test Data ---")
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%  (Spam Precision: High precision means legitimate messages are almost never flagged as spam)")
    print(f"Recall:    {rec * 100:.2f}%  (Spam Recall: Percentage of all actual spam messages correctly detected)")
    print(f"F1-Score:  {f1 * 100:.2f}%  (Harmonic mean of Precision and Recall)")

    print("\nConfusion Matrix:")
    print("                Predicted Ham    Predicted Spam")
    print(f"Actual Ham:     {cm[0][0]:<16} {cm[0][1]}")
    print(f"Actual Spam:    {cm[1][0]:<16} {cm[1][1]}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham (Not Spam)", "Spam"]))

    # --- 6. Save Model & Vectorizer ---
    print("\n--- 6. Saving Model Artifacts ---")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"[+] Saved model to:      {MODEL_PATH}")
    print(f"[+] Saved vectorizer to: {VECTORIZER_PATH}")

    # Save metrics to JSON for the Streamlit dashboard
    metrics_summary = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "true_negatives": int(cm[0][0]),
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
        "true_positives": int(cm[1][1])
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[+] Saved metrics summary to: {METRICS_PATH}")

    return model, vectorizer, metrics_summary


def main():
    print("=" * 60)
    print("   SPAM MESSAGE DETECTION SYSTEM - MODEL TRAINING PIPELINE")
    print("=" * 60)
    df = load_or_download_dataset()
    df_clean = preprocess_data(df)
    train_and_evaluate(df_clean)
    print("\n[OK] All steps completed! You can now launch the web app with:")
    print("    streamlit run app.py")


if __name__ == "__main__":
    main()

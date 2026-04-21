"""
preprocess.py
Data loading and preprocessing pipeline for DeepTruth.
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Constants ────────────────────────────────────────────────────────────────
MAX_WORDS   = 10_000
MAX_LEN     = 200
TEST_SIZE   = 0.20
RANDOM_SEED = 42
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")


def clean_text(text: str) -> str:
    """Lowercase, strip special characters, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)                 # keep letters only
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_raw_data() -> pd.DataFrame:
    """
    Read Fake.csv and True.csv from data/, assign labels,
    return a shuffled DataFrame with columns [text, label].
    """
    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    true_path = os.path.join(DATA_DIR, "True.csv")

    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        raise FileNotFoundError(
            "Fake.csv / True.csv not found in data/.\n"
            "Please follow data/README_dataset.txt to download the ISOT dataset."
        )

    fake_df = pd.read_csv(fake_path, usecols=["text"])
    true_df = pd.read_csv(true_path, usecols=["text"])

    fake_df["label"] = 0   # Fake = 0
    true_df["label"] = 1   # Real = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.dropna(subset=["text"])
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


def preprocess(df: pd.DataFrame = None):
    """
    Full pipeline: clean → tokenize → pad → split.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    tokenizer                         : fitted Keras Tokenizer
    """
    if df is None:
        df = load_raw_data()

    print(f"[preprocess] Total samples: {len(df)}")
    print(f"[preprocess] Label distribution:\n{df['label'].value_counts()}\n")

    # ── Clean text ────────────────────────────────────────────────────────────
    df["text"] = df["text"].apply(clean_text)

    # ── Tokenize ──────────────────────────────────────────────────────────────
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["text"])

    sequences = tokenizer.texts_to_sequences(df["text"])

    # ── Pad ───────────────────────────────────────────────────────────────────
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
    y = df["label"].values.astype(np.float32)

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    print(f"[preprocess] Train: {X_train.shape} | Test: {X_test.shape}")

    # ── Persist tokenizer ────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    tok_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
    with open(tok_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"[preprocess] Tokenizer saved → {tok_path}")

    return X_train, X_test, y_train, y_test, tokenizer


def load_tokenizer():
    """Load persisted tokenizer from models/tokenizer.pkl."""
    tok_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
    if not os.path.exists(tok_path):
        raise FileNotFoundError(
            "tokenizer.pkl not found. Run train.py first."
        )
    with open(tok_path, "rb") as f:
        return pickle.load(f)


def encode_texts(texts: list, tokenizer: Tokenizer) -> np.ndarray:
    """Encode and pad a list of raw text strings for inference."""
    cleaned   = [clean_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned)
    return pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")


if __name__ == "__main__":
    preprocess()

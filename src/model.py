"""
model.py
Defines the LSTM-based DeepTruth model architecture.
"""

import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

MAX_WORDS  = 10_000
OUTPUT_DIM = 128
LSTM_UNITS = 64
DROPOUT    = 0.5
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.h5")


def build_model(max_words: int = MAX_WORDS,
                output_dim: int = OUTPUT_DIM,
                lstm_units: int = LSTM_UNITS,
                dropout: float = DROPOUT,
                learning_rate: float = 1e-3) -> Sequential:
    """
    Build and compile the LSTM classifier.

    Architecture
    ────────────
    Embedding(10 000, 128)
    LSTM(64)
    Dropout(0.5)
    Dense(1, sigmoid)
    """
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=output_dim, name="embedding"),
        LSTM(lstm_units, name="lstm"),
        Dropout(dropout, name="dropout"),
        Dense(1, activation="sigmoid", name="output"),
    ], name="DeepTruth_LSTM")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def save_model(model: Sequential, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"[model] Saved → {path}")


def load_trained_model(path: str = MODEL_PATH) -> Sequential:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Run train.py first."
        )
    model = load_model(path)
    print(f"[model] Loaded ← {path}")
    return model


if __name__ == "__main__":
    m = build_model()
    print("Model built successfully.")

"""
train.py
Training loop for DeepTruth with accuracy/loss plots.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow imports from src/ when run directly
sys.path.insert(0, os.path.dirname(__file__))

from preprocess import preprocess
from model import build_model, save_model

EPOCHS     = 10
BATCH_SIZE = 64
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "plots")


def plot_history(history, plots_dir: str = PLOTS_DIR) -> None:
    """Save accuracy and loss curves as PNG files."""
    os.makedirs(plots_dir, exist_ok=True)

    epochs = range(1, len(history.history["accuracy"]) + 1)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history["accuracy"],     label="Train Accuracy")
    plt.plot(epochs, history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    acc_path = os.path.join(plots_dir, "accuracy_plot.png")
    plt.savefig(acc_path)
    plt.close()
    print(f"[train] Accuracy plot saved → {acc_path}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history["loss"],     label="Train Loss")
    plt.plot(epochs, history.history["val_loss"], label="Val Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(plots_dir, "loss_plot.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"[train] Loss plot saved → {loss_path}")


def train(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
    """
    End-to-end training:
    1. Load & preprocess data
    2. Build model
    3. Fit with validation split
    4. Save model + plots
    """
    # ── Data ──────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, _ = preprocess()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model()

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n[train] Starting training — epochs={epochs}, batch={batch_size}\n")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    # ── Save artefacts ────────────────────────────────────────────────────────
    save_model(model)
    plot_history(history)

    # ── Quick summary ─────────────────────────────────────────────────────────
    val_acc  = max(history.history["val_accuracy"])
    val_loss = min(history.history["val_loss"])
    print(f"\n[train] Best val accuracy : {val_acc:.4f}")
    print(f"[train] Best val loss     : {val_loss:.4f}")

    return model, history, X_test, y_test


if __name__ == "__main__":
    train()

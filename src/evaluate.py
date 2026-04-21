"""
evaluate.py
Evaluation module: metrics + confusion matrix for DeepTruth.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess
from model import load_trained_model

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
CM_PATH     = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")


def evaluate(model=None, X_test=None, y_test=None) -> dict:
    """
    Load model (if not provided), run predictions on test set,
    print full classification report, save confusion matrix PNG.

    Returns a dict with accuracy, precision, recall, f1.
    """
    # ── Load data if not passed in ────────────────────────────────────────────
    if X_test is None or y_test is None:
        _, X_test, _, y_test, _ = preprocess()

    # ── Load model if not passed in ───────────────────────────────────────────
    if model is None:
        model = load_trained_model()

    # ── Predict ───────────────────────────────────────────────────────────────
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 50)
    print("  DeepTruth — Evaluation Results")
    print("=" * 50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print("=" * 50)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Fake (0)", "Real (1)"]))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Fake", "Real"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("DeepTruth — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    plt.close()
    print(f"[evaluate] Confusion matrix saved → {CM_PATH}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


if __name__ == "__main__":
    evaluate()

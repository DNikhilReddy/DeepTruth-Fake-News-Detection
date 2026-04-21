"""
interpret.py
LIME-based interpretability for DeepTruth predictions.
Shows which words most influenced a Fake / Real classification.
"""

import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import clean_text, load_tokenizer, encode_texts, MAX_LEN
from model import load_trained_model

try:
    from lime.lime_text import LimeTextExplainer
except ImportError:
    raise ImportError(
        "LIME is required for interpretability.\n"
        "Install it with:  pip install lime"
    )

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
PLOTS_DIR   = os.path.join(OUTPUTS_DIR, "plots")


# ── Prediction wrapper ────────────────────────────────────────────────────────

def make_predict_fn(model, tokenizer):
    """
    Returns a function compatible with LimeTextExplainer:
    input  → list[str]
    output → np.ndarray shape (n, 2)  [prob_fake, prob_real]
    """
    def predict_proba(texts):
        X    = encode_texts(texts, tokenizer)
        prob = model.predict(X, verbose=0).ravel()          # P(Real)
        return np.column_stack([1 - prob, prob])            # [P(Fake), P(Real)]
    return predict_proba


# ── Main explain function ─────────────────────────────────────────────────────

def explain(text: str,
            num_features: int = 10,
            num_samples: int = 500,
            save_html: bool = True) -> None:
    """
    Generate a LIME explanation for a single article text.

    Parameters
    ----------
    text         : raw news article text
    num_features : number of words to highlight
    num_samples  : LIME perturbation samples (higher = more stable)
    save_html    : if True, save HTML explanation to outputs/
    """
    model     = load_trained_model()
    tokenizer = load_tokenizer()
    predict   = make_predict_fn(model, tokenizer)

    explainer = LimeTextExplainer(class_names=["Fake", "Real"])

    print("[interpret] Running LIME explanation …")
    exp = explainer.explain_instance(
        text,
        predict,
        num_features=num_features,
        num_samples=num_samples,
    )

    # ── Console output ────────────────────────────────────────────────────────
    probs     = predict([text])[0]
    label_idx = int(probs[1] >= 0.5)
    label_str = "REAL" if label_idx == 1 else "FAKE"

    print("\n" + "=" * 60)
    print(f"  Prediction  : {label_str}")
    print(f"  Confidence  : {max(probs):.2%}")
    print("=" * 60)
    print(f"  Top {num_features} influential words:")
    print("-" * 60)
    for word, weight in exp.as_list():
        direction = "→ REAL" if weight > 0 else "→ FAKE"
        print(f"  {word:<25} weight={weight:+.4f}  {direction}")
    print("=" * 60 + "\n")

    # ── Save HTML ─────────────────────────────────────────────────────────────
    if save_html:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        html_path = os.path.join(OUTPUTS_DIR, "lime_explanation.html")
        exp.save_to_file(html_path)
        print(f"[interpret] HTML explanation saved → {html_path}")


# ── CLI demo ──────────────────────────────────────────────────────────────────

DEMO_FAKE = (
    "Breaking: Scientists confirm that 5G towers are secretly being used "
    "to control the population. Government whistleblowers reveal shocking "
    "evidence of mass surveillance and mind-control experiments funded by "
    "shadowy elite organizations."
)

DEMO_REAL = (
    "The Federal Reserve raised interest rates by 25 basis points on Wednesday, "
    "citing continued progress on inflation. Fed Chair Jerome Powell stated that "
    "the decision reflects the committee's commitment to returning inflation "
    "to the 2 percent target over time."
)

if __name__ == "__main__":
    print("\n── DEMO: Fake news example ──────────────────────────────")
    explain(DEMO_FAKE, num_features=10, num_samples=300)

    print("\n── DEMO: Real news example ──────────────────────────────")
    explain(DEMO_REAL, num_features=10, num_samples=300)

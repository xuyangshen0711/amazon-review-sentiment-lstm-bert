"""
CS6120 Final Project - BiLSTM Sentiment Analysis Demo
Author: Xuyang Shen

==============================================================
DESCRIPTION
==============================================================
Standalone demo script for the BiLSTM sentiment classifier.
Loads the pre-trained model (model.pt) and predicts the
sentiment (Positive / Negative) of Amazon-style review text.

No training required — the saved model weights are loaded directly.
The vocabulary and model configuration are also stored in model.pt,
so no dataset or GloVe files are needed at inference time.

NOTE: model.pt must have been produced by lstm_pipeline_weighted.py
after the vocab-saving update. If you have an older model.pt that
lacks the 'vocab' key, re-run the training pipeline once to regenerate it.

==============================================================
REQUIREMENTS
==============================================================
    pip install torch numpy nltk

    python3 -c "import nltk; nltk.download('stopwords')"

==============================================================
HOW TO RUN
==============================================================
    cd bilstm_weighted
    python3 demo.py

The script predicts sentiment for the built-in sample reviews
and then enters an interactive prompt so you can type your own.
==============================================================
"""

import string
import torch
import torch.nn as nn
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH   = "model.pt"
MAX_LENGTH   = 256
LABELS       = {0: "Negative", 1: "Positive"}

STOP_WORDS = set(stopwords.words("english"))

# ── Sample reviews for demonstration ─────────────────────────────────────────
SAMPLE_REVIEWS = [
    # Expected: Positive
    "This software is absolutely fantastic. It solved all my problems "
    "and the interface is intuitive. Highly recommend to everyone!",

    # Expected: Negative
    "Terrible product. It crashed my computer twice and customer support "
    "was completely useless. Total waste of money.",

    # Expected: Negative (negation / sarcasm — challenging for LSTM)
    "I wish I could give this 0 stars. Bought it hoping it would work "
    "as advertised but it does absolutely nothing it promises.",

    # Expected: Positive
    "Works exactly as described. Installation was straightforward and "
    "I haven't had a single issue in three months of daily use.",

    # Expected: Negative (long complaint narrative — challenging for LSTM)
    "I bought this software hoping it would solve my workflow problems. "
    "The first week seemed promising, but then the bugs started appearing. "
    "Files corrupted, autosave stopped working, and after a major update "
    "the whole application refused to launch. Completely unusable.",
]


# ── Model definition (must match lstm_pipeline_weighted.py exactly) ───────────
class BiLSTMWeightedSentiment(nn.Module):
    """Bidirectional LSTM sentiment classifier with max pooling over time steps."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 embedding_matrix, num_layers=2, dropout_prob=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, _ = self.lstm(embedded)
        pooled, _ = outputs.max(dim=1)
        pooled = self.dropout(pooled)
        return self.fc(pooled).squeeze(-1)


# ── Text preprocessing (matches the training pipeline) ───────────────────────
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [token for token in tokens if token not in STOP_WORDS]


def tokens_to_indices(tokens, vocab, max_len):
    indices = [vocab.get(word, vocab["<unk>"]) for word in tokens]
    if len(indices) < max_len:
        indices += [vocab["<pad>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(model_path, device):
    """Load the fine-tuned BiLSTM model and vocabulary from a checkpoint file."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if "vocab" not in checkpoint:
        raise KeyError(
            "'vocab' key not found in model.pt. "
            "Please re-run lstm_pipeline_weighted.py to regenerate the checkpoint "
            "with vocabulary information included."
        )

    vocab        = checkpoint["vocab"]
    hidden_dim   = checkpoint.get("hidden_dim",   128)
    num_layers   = checkpoint.get("num_layers",   2)
    embedding_dim = checkpoint.get("embedding_dim", 300)
    dropout_prob = checkpoint.get("dropout_prob", 0.5)
    threshold    = float(checkpoint["best_threshold"])

    # Build the model architecture; embedding weights come from the state dict
    dummy_matrix = np.zeros((len(vocab), embedding_dim), dtype=np.float32)
    model = BiLSTMWeightedSentiment(
        len(vocab), embedding_dim, hidden_dim,
        dummy_matrix, num_layers, dropout_prob,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, vocab, threshold


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(text, model, vocab, threshold, device, max_length=MAX_LENGTH):
    """
    Predict sentiment for a single review string.

    Returns:
        label      (str):   'Positive' or 'Negative'
        confidence (float): sigmoid probability of the predicted class (0–1)
    """
    tokens  = clean_text(text)
    indices = tokens_to_indices(tokens, vocab, max_length)
    tensor  = torch.tensor([indices], dtype=torch.long).to(device)

    with torch.no_grad():
        logit = model(tensor)
        prob  = torch.sigmoid(logit).item()

    predicted_class = 1 if prob >= threshold else 0
    confidence      = prob if predicted_class == 1 else 1.0 - prob
    return LABELS[predicted_class], confidence


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print("  BiLSTM Sentiment Analysis — Amazon Reviews Demo")
    print("=" * 60)
    print(f"  Device : {device}")
    print(f"  Model  : Bidirectional LSTM  (weighted / fine-tuned)")
    print(f"  Weights: {MODEL_PATH}")
    print("=" * 60)

    print("\nLoading model...", end=" ", flush=True)
    model, vocab, threshold = load_model(MODEL_PATH, device)
    print("done.")
    print(f"  Vocab size       : {len(vocab):,}")
    print(f"  Best threshold   : {threshold:.3f}\n")

    # ── Built-in sample predictions ───────────────────────────────────────────
    print("─" * 60)
    print("SAMPLE PREDICTIONS")
    print("─" * 60)

    for i, review in enumerate(SAMPLE_REVIEWS, start=1):
        label, confidence = predict(review, model, vocab, threshold, device)
        print(f"\n[{i}] {review}")
        print(f"    => Prediction : {label}  (confidence: {confidence:.2%})")

    # ── Interactive mode ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("INTERACTIVE MODE  (type 'quit' to exit)")
    print("─" * 60)

    while True:
        print()
        user_input = input("Enter a review: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Exiting demo.")
            break
        if not user_input:
            continue
        label, confidence = predict(user_input, model, vocab, threshold, device)
        print(f"=> Prediction : {label}  (confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()

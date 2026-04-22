"""
CS6120 Final Project - Simple Weighted LSTM Sentiment Analysis Demo
Author: Xuyang Shen

This is the demo for the simple weighted LSTM classifier.
This model uses class-weighted loss with the same unidirectional
LSTM architecture as the baseline model.
It does not use balanced sampling or threshold tuning.

No training is needed here because the saved model is loaded directly.
The vocabulary and model settings are stored in model.pt, so no
dataset or GloVe files are needed when running the demo.

NOTE: model.pt must have been produced by lstm_pipeline_simple_weighted.py.
If you have not yet trained this model, run:
    cd lstm_simple_weighted
    python3 lstm_pipeline_simple_weighted.py


REQUIREMENTS

    pip install torch numpy nltk

    python3 -c "import nltk; nltk.download('stopwords')"


HOW TO RUN
    cd lstm_simple_weighted
    python3 demo.py
"""

import string
import torch
import torch.nn as nn
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

#  Configuration
MODEL_PATH = "model.pt"
MAX_LENGTH = 256
LABELS     = {0: "Negative", 1: "Positive"}

STOP_WORDS = set(stopwords.words("english"))

#  Sample reviews for demonstration
SAMPLE_REVIEWS = [
    # Expected: Positive
    "This software is absolutely fantastic. It solved all my problems "
    "and the interface is intuitive. Highly recommend to everyone!",

    # Expected: Negative
    "Terrible product. It crashed my computer twice and customer support "
    "was completely useless. Total waste of money.",

    # Expected: Negative
    "I wish I could give this 0 stars. Bought it hoping it would work "
    "as advertised but it does absolutely nothing it promises.",

    # Expected: Positive
    "Works exactly as described. Installation was straightforward and "
    "I haven't had a single issue in three months of daily use.",

    # Expected: Negative
    "I bought this software hoping it would solve my workflow problems. "
    "The first week seemed promising, but then the bugs started appearing. "
    "Files corrupted, autosave stopped working, and after a major update "
    "the whole application refused to launch. Completely unusable.",
]


#  Model definition
class LSTMSentiment(nn.Module):
    """Unidirectional 2-layer LSTM classifier using the final hidden state."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 embedding_matrix, num_layers=2, dropout_prob=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )
        # keep pretrained embeddings fixed during inference
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc      = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        out = self.dropout(hidden[-1, :, :])
        return self.fc(out).squeeze(-1)


#  Text preprocessing
def clean_text(text):
    # lowercase text, remove punctuation, then remove stopwords
    text   = str(text).lower()
    text   = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in STOP_WORDS]


def tokens_to_indices(tokens, vocab, max_len):
    # convert tokens to ids, then pad or truncate to a fixed length
    indices = [vocab.get(word, vocab["<unk>"]) for word in tokens]
    if len(indices) < max_len:
        indices += [vocab["<pad>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices


#  Model loading
def load_model(model_path, device):
    # load saved checkpoint from disk
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if "vocab" not in checkpoint:
        raise KeyError(
            "'vocab' key not found in model.pt. "
            "Please re-run lstm_pipeline_simple_weighted.py to regenerate "
            "the checkpoint with vocabulary information included."
        )

    # read model settings from the checkpoint
    vocab         = checkpoint["vocab"]
    hidden_dim    = checkpoint.get("hidden_dim",    256)
    num_layers    = checkpoint.get("num_layers",    2)
    embedding_dim = checkpoint.get("embedding_dim", 300)
    dropout_prob  = checkpoint.get("dropout_prob",  0.5)
    threshold     = float(checkpoint.get("best_threshold", 0.5))

    # rebuild model structure before loading saved weights
    dummy_matrix = np.zeros((len(vocab), embedding_dim), dtype=np.float32)
    model = LSTMSentiment(
        len(vocab), embedding_dim, hidden_dim,
        dummy_matrix, num_layers, dropout_prob,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, vocab, threshold


#  Inference
def predict(text, model, vocab, threshold, device, max_length=MAX_LENGTH):
    # preprocess input text and convert it to a tensor
    tokens  = clean_text(text)
    indices = tokens_to_indices(tokens, vocab, max_length)
    tensor  = torch.tensor([indices], dtype=torch.long).to(device)

    # run prediction without gradient computation
    with torch.no_grad():
        logit = model(tensor)
        prob  = torch.sigmoid(logit).item()

    # convert probability to binary class using the threshold
    predicted_class = 1 if prob >= threshold else 0
    confidence      = prob if predicted_class == 1 else 1.0 - prob
    return LABELS[predicted_class], confidence


#  Main
def main():
    # choose available device automatically
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    print("  LSTM Sentiment Analysis — Simple Weighted Demo")
    print(f"  Device : {device}")
    print(f"  Model  : Unidirectional LSTM  (simple weighted)")
    print(f"  Weights: {MODEL_PATH}")

    print("\nLoading model...", end=" ", flush=True)
    model, vocab, threshold = load_model(MODEL_PATH, device)
    print("done.")
    print(f"  Vocab size : {len(vocab):,}\n")

    # run a few built-in demo examples first
    print("SAMPLE PREDICTIONS")

    for i, review in enumerate(SAMPLE_REVIEWS, start=1):
        label, confidence = predict(review, model, vocab, threshold, device)
        print(f"\n[{i}] {review}")
        print(f"    => Prediction : {label}  (confidence: {confidence:.2%})")

    # allow the user to test custom input
    print("\n")
    print("INTERACTIVE MODE  (type 'quit' to exit)")

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

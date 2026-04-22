"""
CS6120 Final Project - Simple Weighted BiLSTM Pipeline
Author: Xuyang Shen

==============================================================
METHOD OVERVIEW
==============================================================
Intermediate model between unweighted (V1) and full weighted (V3):
  - Bidirectional LSTM + max pooling  (same architecture as V3)
  - Class-weighted BCEWithLogitsLoss  (pos_weight to penalise FN)
  - Standard DataLoader               (NO WeightedRandomSampler)
  - Fixed 0.5 threshold               (no threshold tuning)
==============================================================
"""

import os
import gzip
import json
import string
import time
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

nltk.download("stopwords", quiet=True)

torch.manual_seed(42)
np.random.seed(42)

STOP_WORDS = set(stopwords.words("english"))
OUTPUT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_ROOTS = (
    OUTPUT_DIR,
    OUTPUT_DIR.parent,
    OUTPUT_DIR.parent.parent / "CS6120" / "Presentation",
    Path.home() / "Desktop" / "CS6120" / "Presentation",
    Path.cwd(),
)


def resolve_resource(name, expect_dir=False):
    for root in SEARCH_ROOTS:
        candidate = root / name
        if expect_dir and candidate.is_dir():
            return candidate
        if not expect_dir and candidate.is_file():
            return candidate
    checked = "\n".join(f"  - {r / name}" for r in SEARCH_ROOTS)
    raise FileNotFoundError(f"Could not find '{name}'. Checked:\n{checked}")


# ── Hyperparameters ────────────────────────────────────────────────────────────
MAX_SEQ_LENGTH = 256
HIDDEN_DIM     = 128
EMBEDDING_DIM  = 300
NUM_LAYERS     = 2
DROPOUT_PROB   = 0.5
BATCH_SIZE     = 64
EPOCHS         = 10
PATIENCE       = 3
LR             = 1e-3

MODEL_PATH = OUTPUT_DIR / "model.pt"


# ── Data loading ───────────────────────────────────────────────────────────────
def load_amazon_data(file_path, max_reviews=100000):
    data = []
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_reviews:
                break
            record = json.loads(line)
            if "reviewText" in record and "overall" in record:
                data.append({"text": record["reviewText"], "rating": float(record["overall"])})
    return pd.DataFrame(data)


# ── Preprocessing ──────────────────────────────────────────────────────────────
def clean_text(text):
    text   = str(text).lower()
    text   = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in STOP_WORDS]


def build_vocab(sentences, min_freq=2):
    counts = Counter(word for tokens in sentences for word in tokens)
    vocab  = {"<pad>": 0, "<unk>": 1}
    for word, freq in counts.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def load_glove_embeddings(glove_path):
    embeddings = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word  = parts[0]
            try:
                embeddings[word] = np.array(parts[1:], dtype=np.float32)
            except ValueError:
                continue
    return embeddings


def create_embedding_matrix(vocab, embeddings_dict, embedding_dim=300):
    matrix = np.zeros((len(vocab), embedding_dim), dtype=np.float32)
    for word, idx in vocab.items():
        if word in embeddings_dict:
            matrix[idx] = embeddings_dict[word]
        elif word not in ("<pad>", "<unk>"):
            matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return matrix


def tokens_to_indices(tokens, vocab, max_len):
    indices = [vocab.get(w, vocab["<unk>"]) for w in tokens]
    if len(indices) < max_len:
        indices += [vocab["<pad>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices


# ── Dataset ────────────────────────────────────────────────────────────────────
class AmazonReviewDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.sequences = df["tokens"].apply(
            lambda t: tokens_to_indices(t, vocab, max_len)
        ).tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx],    dtype=torch.float),
        )


# ── Model ──────────────────────────────────────────────────────────────────────
class BiLSTMSimpleWeighted(nn.Module):
    """BiLSTM + max pooling — same architecture as V3, trained without sampler."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 embedding_matrix, num_layers=2, dropout_prob=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc      = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded      = self.embedding(x)
        outputs, _    = self.lstm(embedded)
        pooled, _     = outputs.max(dim=1)
        pooled        = self.dropout(pooled)
        return self.fc(pooled).squeeze(-1)


# ── Main pipeline ──────────────────────────────────────────────────────────────
def main():
    data_path  = resolve_resource("Software_5.json.gz")
    glove_path = resolve_resource("glove_data/glove.6B.300d.txt", expect_dir=False)

    print(f"Dataset : {data_path}")
    print(f"GloVe   : {glove_path}")

    # Load & label
    df = load_amazon_data(data_path)
    df["label"]  = (df["rating"] >= 4).astype(int)
    df["tokens"] = df["text"].apply(clean_text)
    df = df[df["tokens"].map(len) > 0].reset_index(drop=True)

    neg = (df["label"] == 0).sum()
    pos = (df["label"] == 1).sum()
    print(f"Class distribution — Negative: {neg}  Positive: {pos}")

    train_df, temp_df = train_test_split(df,      test_size=0.2,  random_state=42, stratify=df["label"])
    val_df,   test_df = train_test_split(temp_df, test_size=0.5,  random_state=42, stratify=temp_df["label"])
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    vocab = build_vocab(train_df["tokens"])
    print(f"Vocabulary size: {len(vocab)}")

    glove_dict       = load_glove_embeddings(glove_path)
    embedding_matrix = create_embedding_matrix(vocab, glove_dict, EMBEDDING_DIM)

    train_ds = AmazonReviewDataset(train_df, vocab, MAX_SEQ_LENGTH)
    val_ds   = AmazonReviewDataset(val_df,   vocab, MAX_SEQ_LENGTH)
    test_ds  = AmazonReviewDataset(test_df,  vocab, MAX_SEQ_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)  # no sampler (unlike V3)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model = BiLSTMSimpleWeighted(
        len(vocab), EMBEDDING_DIM, HIDDEN_DIM, embedding_matrix, NUM_LAYERS, DROPOUT_PROB
    ).to(device)

    neg_count  = (train_df["label"] == 0).sum()
    pos_count  = (train_df["label"] == 1).sum()
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)
    print(f"pos_weight: {pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    best_val_loss   = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        # Training
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)
        train_losses.append(avg_train)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                val_loss += criterion(model(inputs), labels).item()
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab":            vocab,
                "hidden_dim":       HIDDEN_DIM,
                "num_layers":       NUM_LAYERS,
                "embedding_dim":    EMBEDDING_DIM,
                "dropout_prob":     DROPOUT_PROB,
                "best_threshold":   0.5,
            }, MODEL_PATH)
            print("  -> Saved.")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered!")
                break

    # Evaluation
    ck = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs)
            targets.extend(labels.numpy())

    preds_binary = [1 if p >= 0.5 else 0 for p in preds]
    acc  = accuracy_score(targets, preds_binary)
    prec, rec, f1, _ = precision_recall_fscore_support(targets, preds_binary, average="binary")
    _, _, macro_f1, _ = precision_recall_fscore_support(targets, preds_binary, average="macro")
    neg_prec, neg_rec, neg_f1, _ = precision_recall_fscore_support(
        targets, preds_binary, average=None, labels=[0]
    )
    cm = confusion_matrix(targets, preds_binary)

    results = [
        "=" * 60,
        "TEST METRICS",
        "=" * 60,
        f"Model: BiLSTM (Simple Weighted)",
        f"Accuracy:           {acc:.4f}",
        f"Precision:          {prec:.4f}",
        f"Recall:             {rec:.4f}",
        f"F1-Score:           {f1:.4f}",
        f"Macro F1:           {macro_f1:.4f}",
        f"Negative Precision: {neg_prec[0]:.4f}",
        f"Negative Recall:    {neg_rec[0]:.4f}",
        f"Negative F1:        {neg_f1[0]:.4f}",
        f"Confusion Matrix:\n{cm}",
    ]
    report = "\n".join(results)
    print(report)
    (OUTPUT_DIR / "test_metrics.txt").write_text(report)

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.title("Simple Weighted BiLSTM — Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curves.png")
    plt.close()

    # Confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap="Blues")
    plt.title("Simple Weighted BiLSTM — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()

    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time() - t0:.1f}s")

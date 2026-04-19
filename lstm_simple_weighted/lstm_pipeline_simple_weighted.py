"""
CS6120 Final Project - LSTM Pipeline (Simple Weighted Loss)
Author: Xuyang Shen

==============================================================
WHAT CHANGED vs. UNWEIGHTED
==============================================================
EXACTLY ONE change from the unweighted baseline:
  - Loss function: BCELoss (unweighted)
                -> BCEWithLogitsLoss(pos_weight=num_neg/num_pos)

Everything else is IDENTICAL:
  - Same unidirectional LSTM architecture
  - Same hidden_dim = 256
  - Same GloVe 300d frozen embeddings
  - Same data split (random_state=42)
  - Same optimizer (Adam, lr=1e-3)
  - Same early stopping (patience=3, monitor val_loss)
  - Same threshold (logit > 0 ↔ sigmoid(logit) > 0.5)
  - Same max_seq_length = 256, batch_size = 64

This is a single-variable controlled experiment.
==============================================================

HOW TO RUN
==============================================================
  python3 lstm_pipeline_simple_weighted.py

Output files (all saved in this folder):
  model.pt
  loss_curves.png
  confusion_matrix.png
  test_metrics.txt
  model_summary.txt
  comparison_to_unweighted.txt  (auto-generated if baseline exists)
==============================================================
"""

import os
import gzip
import json
import string
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from collections import Counter
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Reproducibility — identical seeds to unweighted version
torch.manual_seed(42)
np.random.seed(42)

STOP_WORDS = set(stopwords.words('english'))
OUTPUT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_ROOTS = (
    OUTPUT_DIR,
    OUTPUT_DIR.parent,
    OUTPUT_DIR.parent / "CS6120" / "Presentation",
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
    kind = "directory" if expect_dir else "file"
    raise FileNotFoundError(f"Could not find {kind} '{name}'. Checked:\n{checked}")


def save_text_report(path, lines):
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


DATA_PATH  = resolve_resource("Software_5.json.gz")
GLOVE_PATH = resolve_resource("glove_data", expect_dir=True) / "glove.6B.300d.txt"
BASELINE_METRICS_PATH = OUTPUT_DIR.parent / "lstm_unweighted" / "test_metrics.txt"

MODEL_PATH          = OUTPUT_DIR / "model.pt"
LOSS_CURVES_PATH    = OUTPUT_DIR / "loss_curves.png"
CM_PATH             = OUTPUT_DIR / "confusion_matrix.png"
TEST_METRICS_PATH   = OUTPUT_DIR / "test_metrics.txt"
SUMMARY_PATH        = OUTPUT_DIR / "model_summary.txt"
COMPARISON_PATH     = OUTPUT_DIR / "comparison_to_unweighted.txt"

print(f"Artifacts will be saved to: {OUTPUT_DIR}")
print(f"Using dataset:              {DATA_PATH}")
print(f"Using GloVe embeddings:     {GLOVE_PATH}")


# ── SECTION 1: DATA ──────────────────────────────────────────

def load_amazon_data(file_path, max_reviews=100000):
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= max_reviews:
                break
            record = json.loads(line)
            if 'reviewText' in record and 'overall' in record:
                data.append({'text': record['reviewText'],
                             'rating': float(record['overall'])})
    return pd.DataFrame(data)

df = load_amazon_data(DATA_PATH, max_reviews=100000)
df = df[df['rating'] != 3.0]
df['label'] = df['rating'].apply(lambda x: 1 if x > 3.0 else 0)

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return [t for t in text.split() if t not in STOP_WORDS]

df['tokens'] = df['text'].apply(clean_text)

# Identical split to unweighted version
train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
train_df,  val_df    = train_test_split(train_val_df, test_size=1/9, random_state=42,
                                         stratify=train_val_df['label'])

print(f"\nDataset loaded (after excluding 3-star): {len(df)} reviews")
print(f"  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
print(f"  Positive ratio: {df['label'].mean():.2%}")

# ── CLASS WEIGHTS (the only new thing) ───────────────────────
num_pos = int(train_df['label'].sum())
num_neg = int(len(train_df) - num_pos)
# pos_weight < 1 de-emphasises the majority (positive) class,
# so the minority (negative) class carries more relative influence.
# Formula: pos_weight = num_neg / num_pos  (standard inverse-frequency)
pos_weight_value = num_neg / num_pos
print(f"\n  Positive samples (train): {num_pos}")
print(f"  Negative samples (train): {num_neg}")
print(f"  pos_weight = {num_neg}/{num_pos} = {pos_weight_value:.4f}")
print(f"  → positive class loss is scaled by {pos_weight_value:.4f}; "
      f"negative class relative weight increases.")


# ── SECTION 2: VOCAB & EMBEDDINGS ────────────────────────────

def build_vocab(sentences, min_freq=2):
    counter = Counter()
    for tokens in sentences:
        counter.update(tokens)
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab

vocab = build_vocab(train_df['tokens'])
print(f"\nVocabulary size: {len(vocab)}")

def load_glove_embeddings(glove_path, expected_dim=300):
    emb = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word   = values[0]
            vector = np.asarray(values[1:], "float32")
            if len(vector) == expected_dim:
                emb[word] = vector
    return emb

glove_dict = load_glove_embeddings(GLOVE_PATH)

def create_embedding_matrix(vocab, emb_dict, embedding_dim=300):
    matrix = np.zeros((len(vocab), embedding_dim))
    for word, i in vocab.items():
        if word in emb_dict:
            matrix[i] = emb_dict[word]
        elif word not in ("<pad>", "<unk>"):
            matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return matrix

embedding_matrix = create_embedding_matrix(vocab, glove_dict)
print(f"Embedding matrix shape: {embedding_matrix.shape}")


# ── SECTION 3: DATASET & DATALOADERS ─────────────────────────

MAX_SEQ_LENGTH = 256
BATCH_SIZE     = 64

def tokens_to_indices(tokens, vocab, max_len):
    indices = [vocab.get(w, vocab["<unk>"]) for w in tokens]
    if len(indices) < max_len:
        indices += [vocab["<pad>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices

class AmazonReviewDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.labels    = df['label'].values
        self.sequences = df['tokens'].apply(
            lambda t: tokens_to_indices(t, vocab, max_len)).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.sequences[idx], dtype=torch.long),
                torch.tensor(self.labels[idx],    dtype=torch.float))

train_dataset = AmazonReviewDataset(train_df, vocab, MAX_SEQ_LENGTH)
val_dataset   = AmazonReviewDataset(val_df,   vocab, MAX_SEQ_LENGTH)
test_dataset  = AmazonReviewDataset(test_df,  vocab, MAX_SEQ_LENGTH)

# Standard random shuffle — identical to unweighted
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"\nDataLoaders — Train: {len(train_loader)} batches  "
      f"Val: {len(val_loader)}  Test: {len(test_loader)}")


# ── SECTION 4: MODEL (identical architecture to unweighted) ──

class LSTMSentiment(nn.Module):
    """
    IDENTICAL to the unweighted model, EXCEPT:
      - forward() returns raw logits (no sigmoid)
      because BCEWithLogitsLoss applies sigmoid internally.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 embedding_matrix, num_layers=2, dropout_prob=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False          # frozen

        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_prob if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc      = nn.Linear(hidden_dim, 1)
        # No sigmoid here — BCEWithLogitsLoss handles it

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        out = self.dropout(hidden[-1, :, :])   # last layer, last step
        return self.fc(out).squeeze()          # raw logit

HIDDEN_DIM = 256   # same as unweighted

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"\nUsing device: {device}")

model = LSTMSentiment(len(vocab), 300, HIDDEN_DIM, embedding_matrix).to(device)
print(f"Model: LSTM (Simple Weighted Loss)")
print(f"  Total parameters:     {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable parameters: "
      f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# ── SECTION 5: TRAINING ───────────────────────────────────────

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# THE ONE CHANGE: BCEWithLogitsLoss with pos_weight
# pos_weight is applied only to positive (label=1) examples.
# Setting it < 1 reduces positive class influence → negative class
# is relatively more important → model corrects more for negative errors.
pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

EPOCHS  = 10
PATIENCE = 3

best_val_loss    = float('inf')
patience_counter = 0
train_losses     = []
val_losses       = []
val_accuracies   = []
training_start   = time.time()

for epoch in range(EPOCHS):
    # ── Train ──
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ── Validate ──
    model.eval()
    val_loss   = 0
    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            val_loss += criterion(logits, labels).item()
            # logit > 0  ↔  sigmoid(logit) > 0.5
            preds = (logits > 0).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_acc = accuracy_score(all_labels, all_preds)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS}  "
          f"Train Loss: {avg_train_loss:.4f}  "
          f"Val Loss: {avg_val_loss:.4f}  "
          f"Val Acc: {val_acc:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print("  -> Validation loss improved. Model saved.")
    else:
        patience_counter += 1
        print(f"  -> No improvement. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered!")
            break

total_training_time = time.time() - training_start
print(f"\nTraining complete! {total_training_time:.1f}s ({total_training_time/60:.1f} min)")


# ── SECTION 6: TEST EVALUATION ───────────────────────────────

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

test_preds, test_targets = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        preds  = (logits > 0).long()
        test_preds.extend(preds.cpu().numpy())
        test_targets.extend(labels.cpu().numpy())

acc                               = accuracy_score(test_targets, test_preds)
precision, recall, f1, _          = precision_recall_fscore_support(
    test_targets, test_preds, average='binary')
prec_per, rec_per, f1_per, sup_per = precision_recall_fscore_support(
    test_targets, test_preds, average=None)
conf_matrix                       = confusion_matrix(test_targets, test_preds)

print(f"\nTest Accuracy:  {acc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall:    {recall:.4f}")
print(f"Test F1-Score:  {f1:.4f}")
print(f"\nPer-class metrics:")
print(f"  Negative — Prec: {prec_per[0]:.4f}  Rec: {rec_per[0]:.4f}  "
      f"F1: {f1_per[0]:.4f}  n={sup_per[0]}")
print(f"  Positive — Prec: {prec_per[1]:.4f}  Rec: {rec_per[1]:.4f}  "
      f"F1: {f1_per[1]:.4f}  n={sup_per[1]}")
print(f"\nConfusion Matrix:\n{conf_matrix}")

metrics_lines = [
    "=" * 60,
    "TEST METRICS",
    "=" * 60,
    "Model: LSTM (Simple Weighted Loss)",
    f"Dataset: {DATA_PATH}",
    f"pos_weight: {pos_weight_value:.4f}  (= num_neg / num_pos = {num_neg}/{num_pos})",
    f"Accuracy: {acc:.4f}",
    f"Precision: {precision:.4f}",
    f"Recall: {recall:.4f}",
    f"F1-Score: {f1:.4f}",
    f"Negative Precision: {prec_per[0]:.4f}",
    f"Negative Recall: {rec_per[0]:.4f}",
    f"Negative F1: {f1_per[0]:.4f}",
    f"Negative Support: {sup_per[0]}",
    f"Positive Precision: {prec_per[1]:.4f}",
    f"Positive Recall: {rec_per[1]:.4f}",
    f"Positive F1: {f1_per[1]:.4f}",
    f"Positive Support: {sup_per[1]}",
    "Confusion Matrix:",
    str(conf_matrix),
]
save_text_report(TEST_METRICS_PATH, metrics_lines)
print(f"\nTest metrics saved to: {TEST_METRICS_PATH}")


# ── SECTION 7: VISUALIZATIONS ────────────────────────────────

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, len(val_losses)   + 1), val_losses,   label='Val Loss',   marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies,
         label='Val Accuracy', marker='o', color='green')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(LOSS_CURVES_PATH, dpi=150)
plt.close()
print(f"Loss curves saved to:     {LOSS_CURVES_PATH}")

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                       display_labels=['Negative', 'Positive']).plot(
    ax=ax, cmap='Blues', values_format='d')
plt.title('LSTM (Simple Weighted) - Confusion Matrix on Test Set')
plt.tight_layout()
plt.savefig(CM_PATH, dpi=150)
plt.close()
print(f"Confusion matrix saved to: {CM_PATH}")


# ── SECTION 8: MODEL SUMMARY ─────────────────────────────────

model_size_mb = sum(p.numel() * p.element_size()
                    for p in model.parameters()) / (1024 * 1024)

sample_inputs, _ = next(iter(test_loader))
sample_inputs    = sample_inputs.to(device)
with torch.no_grad():
    _ = model(sample_inputs)   # warm-up

inference_times = []
with torch.no_grad():
    for _ in range(10):
        t0 = time.time()
        _ = model(sample_inputs)
        inference_times.append(time.time() - t0)
avg_inf = np.mean(inference_times)

summary_lines = [
    "=" * 60,
    "MODEL SUMMARY",
    "=" * 60,
    "  Model:             LSTM (Simple Weighted Loss)",
    f"  Total Parameters:  {sum(p.numel() for p in model.parameters()):,}",
    f"  Model Size:        {model_size_mb:.1f} MB",
    f"  Training Time:     {total_training_time:.1f}s ({total_training_time/60:.1f} min)",
    f"  Avg Inference Time (batch of {BATCH_SIZE}): {avg_inf*1000:.1f} ms",
    f"  pos_weight:        {pos_weight_value:.4f}",
    f"  Test Accuracy:     {acc:.4f}",
    f"  Test F1-Score:     {f1:.4f}",
    f"  Neg Precision:     {prec_per[0]:.4f}",
    f"  Neg Recall:        {rec_per[0]:.4f}",
]
save_text_report(SUMMARY_PATH, summary_lines)
print("\n" + "\n".join(summary_lines))


# ── SECTION 9: COMPARISON AGAINST UNWEIGHTED BASELINE ────────

def parse_baseline(path):
    metrics = {}
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metrics[key.strip()] = value.strip()
    return metrics

baseline = parse_baseline(BASELINE_METRICS_PATH)
if baseline:
    try:
        bl_neg_prec = float(baseline["Negative Precision"])
        bl_neg_rec  = float(baseline["Negative Recall"])
        bl_pos_f1   = float(baseline["F1-Score"])

        neg_prec_delta = prec_per[0] - bl_neg_prec
        neg_rec_delta  = rec_per[0]  - bl_neg_rec
        pos_f1_delta   = f1          - bl_pos_f1

        meets_prec = abs(neg_prec_delta) >= 0.10
        meets_rec  = abs(neg_rec_delta)  >= 0.10

        comparison_lines = [
            "=" * 60,
            "COMPARISON: Simple Weighted vs. Unweighted",
            "=" * 60,
            "",
            "NEGATIVE CLASS (minority — the class that matters most)",
            f"  Precision:  {bl_neg_prec:.4f} → {prec_per[0]:.4f}  "
            f"({neg_prec_delta:+.4f})  "
            f"{'✓ >=10% change' if meets_prec else '✗ <10% change'}",
            f"  Recall:     {bl_neg_rec:.4f} → {rec_per[0]:.4f}  "
            f"({neg_rec_delta:+.4f})  "
            f"{'✓ >=10% change' if meets_rec else '✗ <10% change'}",
            "",
            "POSITIVE CLASS",
            f"  F1:         {bl_pos_f1:.4f} → {f1:.4f}  "
            f"({pos_f1_delta:+.4f})",
            f"  Recall:     {float(baseline.get('Recall', 'nan')):.4f} → {recall:.4f}  "
            f"({recall - float(baseline.get('Recall', 0)):+.4f})",
            "",
            "REQUIREMENT CHECK",
            f"  Professor requires >=10% improvement in precision OR recall.",
            f"  Negative Precision: {'PASS' if meets_prec else 'FAIL'}",
            f"  Negative Recall:    {'PASS' if meets_rec  else 'FAIL'}",
        ]
        comparison_text = save_text_report(COMPARISON_PATH, comparison_lines)
        print("\n" + comparison_text, end="")
        print(f"Comparison saved to: {COMPARISON_PATH}")
    except (KeyError, ValueError) as e:
        print(f"[Warning] Could not parse baseline metrics: {e}")
else:
    print(f"\n[Info] Baseline not found at {BASELINE_METRICS_PATH}. "
          f"Skipping comparison.")

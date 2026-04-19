"""
CS6120 Final Project - LSTM Pipeline (Weighted / Improved)
Author: Xuyang Shen

==============================================================
INSTALLATION
==============================================================
Install required dependencies with:
    pip install torch numpy pandas scikit-learn matplotlib nltk

Then download NLTK stopwords (run once):
    python3 -c "import nltk; nltk.download('stopwords')"

==============================================================
HOW TO RUN (Local)
==============================================================
1. Keep Software_5.json.gz and glove_data/ either in this folder
   or in /Users/barry/Desktop/CS6120/Presentation/.
2. Run the full pipeline from this folder:
       python3 lstm_pipeline_weighted.py
3. All outputs are saved into this folder automatically:
       model.pt
       loss_curves.png
       confusion_matrix.png
       test_metrics.txt
       model_summary.txt
       comparison_to_unweighted.txt

==============================================================
METHOD OVERVIEW
==============================================================
This weighted version improves minority-class performance with:
1. Bidirectional LSTM + max pooling
2. Balanced sampling during training
3. Class-weighted BCEWithLogitsLoss
4. Validation-based threshold tuning
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

nltk.download("stopwords", quiet=True)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

STOP_WORDS = set(stopwords.words("english"))
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

    checked_paths = "\n".join(f"  - {root / name}" for root in SEARCH_ROOTS)
    kind = "directory" if expect_dir else "file"
    raise FileNotFoundError(f"Could not find required {kind} '{name}'. Checked:\n{checked_paths}")


def save_text_report(path, lines):
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def parse_metrics_report(path):
    metrics = {}
    if not path.exists():
        return metrics

    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metrics[key.strip()] = value.strip()
    return metrics


DATA_PATH = resolve_resource("Software_5.json.gz")
GLOVE_PATH = resolve_resource("glove_data", expect_dir=True) / "glove.6B.300d.txt"
BASELINE_METRICS_PATH = OUTPUT_DIR.parent / "lstm_unweighted" / "test_metrics.txt"

MODEL_PATH = OUTPUT_DIR / "model.pt"
LOSS_CURVES_PATH = OUTPUT_DIR / "loss_curves.png"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
TEST_METRICS_PATH = OUTPUT_DIR / "test_metrics.txt"
SUMMARY_PATH = OUTPUT_DIR / "model_summary.txt"
COMPARISON_PATH = OUTPUT_DIR / "comparison_to_unweighted.txt"


def load_amazon_data(file_path, max_reviews=100000):
    data = []
    with gzip.open(file_path, "rt", encoding="utf-8") as file_obj:
        for idx, line in enumerate(file_obj):
            if idx >= max_reviews:
                break
            record = json.loads(line)
            if "reviewText" in record and "overall" in record:
                data.append({
                    "text": record["reviewText"],
                    "rating": float(record["overall"]),
                })
    return pd.DataFrame(data)


def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [token for token in tokens if token not in STOP_WORDS]


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


def load_glove_embeddings(glove_path, expected_dim=300):
    embeddings_dict = {}
    with open(glove_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            if len(vector) == expected_dim:
                embeddings_dict[word] = vector
    return embeddings_dict


def create_embedding_matrix(vocab, embeddings_dict, embedding_dim=300):
    matrix = np.zeros((len(vocab), embedding_dim), dtype=np.float32)
    for word, idx in vocab.items():
        if word in embeddings_dict:
            matrix[idx] = embeddings_dict[word]
        elif word not in {"<pad>", "<unk>"}:
            matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,)).astype(np.float32)
    return matrix


MAX_SEQ_LENGTH = 256
BATCH_SIZE = 64


def tokens_to_indices(tokens, vocab, max_len):
    indices = [vocab.get(word, vocab["<unk>"]) for word in tokens]
    if len(indices) < max_len:
        indices += [vocab["<pad>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices


class AmazonReviewDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.labels = df["label"].astype(np.float32).values
        self.sequences = df["tokens"].apply(lambda tokens: tokens_to_indices(tokens, vocab, max_len)).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class BiLSTMWeightedSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, embedding_matrix, num_layers=2, dropout_prob=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
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


def calculate_metrics(labels, positive_predictions):
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels,
        positive_predictions,
        average=None,
        labels=[0, 1],
        zero_division=0,
    )

    confusion = confusion_matrix(labels, positive_predictions)
    accuracy = accuracy_score(labels, positive_predictions)
    macro_f1 = (f1[0] + f1[1]) / 2

    return {
        "accuracy": accuracy,
        "positive_precision": prec[1],
        "positive_recall": rec[1],
        "positive_f1": f1[1],
        "negative_precision": prec[0],
        "negative_recall": rec[0],
        "negative_f1": f1[0],
        "macro_f1": macro_f1,
        "confusion_matrix": confusion,
    }


def collect_probabilities(model, loader, device):
    probabilities = []
    labels = []
    with torch.no_grad():
        for inputs, batch_labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probabilities.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(batch_labels.numpy())
    return np.array(probabilities), np.array(labels).astype(int)


def evaluate_loss(model, loader, criterion, device, negative_weight, positive_weight):
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            batch_loss = criterion(logits, labels)
            batch_weights = torch.where(labels < 0.5, negative_weight, positive_weight)
            total_loss += (batch_loss * batch_weights).mean().item()
    return total_loss / max(1, len(loader))


def search_best_threshold(probabilities, labels):
    best_result = None
    for threshold in np.linspace(0.10, 0.90, 33):
        positive_predictions = (probabilities >= threshold).astype(int)
        metrics = calculate_metrics(labels, positive_predictions)
        result = {
            "threshold": float(threshold),
            "metrics": metrics,
        }

        if best_result is None:
            best_result = result
            continue

        current = metrics["macro_f1"]
        best = best_result["metrics"]["macro_f1"]
        if current > best:
            best_result = result
        elif np.isclose(current, best):
            if metrics["negative_recall"] > best_result["metrics"]["negative_recall"]:
                best_result = result

    return best_result


print(f"Artifacts will be saved to: {OUTPUT_DIR}")
print(f"Using dataset: {DATA_PATH}")
print(f"Using GloVe embeddings: {GLOVE_PATH}")

df = load_amazon_data(DATA_PATH, max_reviews=100000)
df = df[df["rating"] != 3.0]
df["label"] = df["rating"].apply(lambda rating: 1 if rating > 3.0 else 0)
df["tokens"] = df["text"].apply(clean_text)

train_val_df, test_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    stratify=df["label"],
)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=1 / 9,
    random_state=42,
    stratify=train_val_df["label"],
)

print("Dataset loaded successfully!")
print(f"  Total reviews (after excluding 3-star): {len(df)}")
print(f"  Train size: {len(train_df)}")
print(f"  Val size:   {len(val_df)}")
print(f"  Test size:  {len(test_df)}")
print(f"  Positive ratio: {df['label'].mean():.2%}")

class_counts = np.bincount(train_df["label"].astype(int))
negative_count = int(class_counts[0])
positive_count = int(class_counts[1])
negative_weight_value = len(train_df) / (2 * negative_count)
positive_weight_value = len(train_df) / (2 * positive_count)

print("\nTraining distribution:")
print(f"  Negative samples in train: {negative_count}")
print(f"  Positive samples in train: {positive_count}")
print(f"  Negative class weight: {negative_weight_value:.4f}")
print(f"  Positive class weight: {positive_weight_value:.4f}")

vocab = build_vocab(train_df["tokens"])
print(f"Vocabulary size: {len(vocab)}")

glove_dict = load_glove_embeddings(GLOVE_PATH)
embedding_matrix = create_embedding_matrix(vocab, glove_dict)
print(f"Embedding matrix shape: {embedding_matrix.shape}")

train_dataset = AmazonReviewDataset(train_df, vocab, MAX_SEQ_LENGTH)
val_dataset = AmazonReviewDataset(val_df, vocab, MAX_SEQ_LENGTH)
test_dataset = AmazonReviewDataset(test_df, vocab, MAX_SEQ_LENGTH)

sample_weights = np.where(train_df["label"].values == 0, 1.0 / negative_count, 1.0 / positive_count)
balanced_sampler = WeightedRandomSampler(
    weights=torch.as_tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=balanced_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("\nDataLoaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches:   {len(val_loader)}")
print(f"  Test batches:  {len(test_loader)}")

HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT_PROB = 0.5

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"\nUsing device: {device}")
model = BiLSTMWeightedSentiment(
    len(vocab),
    300,
    HIDDEN_DIM,
    embedding_matrix,
    num_layers=NUM_LAYERS,
    dropout_prob=DROPOUT_PROB,
).to(device)

print("\nModel: BiLSTM with Balanced Sampling + Weighted BCE")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=8e-4)
criterion = nn.BCEWithLogitsLoss(reduction="none")
negative_weight = torch.tensor(negative_weight_value, dtype=torch.float32, device=device)
positive_weight = torch.tensor(positive_weight_value, dtype=torch.float32, device=device)

EPOCHS = 10
PATIENCE = 4

train_losses = []
val_losses = []
val_macro_f1_scores = []
val_negative_recalls = []

best_val_macro_f1 = float("-inf")
best_threshold = 0.5
patience_counter = 0
training_start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        batch_loss = criterion(logits, labels)
        batch_weights = torch.where(labels < 0.5, negative_weight, positive_weight)
        loss = (batch_loss * batch_weights).mean()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    avg_val_loss = evaluate_loss(model, val_loader, criterion, device, negative_weight, positive_weight)
    val_losses.append(avg_val_loss)

    val_probabilities, val_labels = collect_probabilities(model, val_loader, device)
    threshold_result = search_best_threshold(val_probabilities, val_labels)
    epoch_threshold = threshold_result["threshold"]
    epoch_metrics = threshold_result["metrics"]

    val_macro_f1_scores.append(epoch_metrics["macro_f1"])
    val_negative_recalls.append(epoch_metrics["negative_recall"])

    print(
        f"Epoch {epoch + 1}/{EPOCHS} - "
        f"Train Loss: {avg_train_loss:.4f} - "
        f"Val Loss: {avg_val_loss:.4f} - "
        f"Val Macro F1: {epoch_metrics['macro_f1']:.4f} - "
        f"Val Neg Recall: {epoch_metrics['negative_recall']:.4f} - "
        f"Threshold: {epoch_threshold:.3f}"
    )

    if epoch_metrics["macro_f1"] > best_val_macro_f1:
        best_val_macro_f1 = epoch_metrics["macro_f1"]
        best_threshold = epoch_threshold
        patience_counter = 0

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "best_threshold": best_threshold,
            "best_val_macro_f1": best_val_macro_f1,
            "negative_weight": negative_weight_value,
            "positive_weight": positive_weight_value,
        }
        torch.save(checkpoint, MODEL_PATH)
        print("  -> Validation macro F1 improved. Model saved.")
    else:
        patience_counter += 1
        print(f"  -> No improvement. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered!")
            break

total_training_time = time.time() - training_start_time
print(f"\nTraining complete! Total time: {total_training_time:.1f}s ({total_training_time / 60:.1f} min)")

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
best_threshold = float(checkpoint["best_threshold"])
best_val_macro_f1 = float(checkpoint["best_val_macro_f1"])
model.eval()

test_probabilities, test_labels = collect_probabilities(model, test_loader, device)
test_positive_predictions = (test_probabilities >= best_threshold).astype(int)
test_metrics = calculate_metrics(test_labels, test_positive_predictions)
conf_matrix = test_metrics["confusion_matrix"]

print(f"\nBest threshold selected from validation: {best_threshold:.3f}")
print(f"Best validation macro F1: {best_val_macro_f1:.4f}")
print(f"\nTest Accuracy:  {test_metrics['accuracy']:.4f}")
print(f"Test Precision: {test_metrics['positive_precision']:.4f}")
print(f"Test Recall:    {test_metrics['positive_recall']:.4f}")
print(f"Test F1-Score:  {test_metrics['positive_f1']:.4f}")

print("\nPer-class metrics:")
print(
    "  Negative - "
    f"Precision: {test_metrics['negative_precision']:.4f}, "
    f"Recall: {test_metrics['negative_recall']:.4f}, "
    f"F1: {test_metrics['negative_f1']:.4f}, "
    f"Support: {np.sum(test_labels == 0)}"
)
print(
    "  Positive - "
    f"Precision: {test_metrics['positive_precision']:.4f}, "
    f"Recall: {test_metrics['positive_recall']:.4f}, "
    f"F1: {test_metrics['positive_f1']:.4f}, "
    f"Support: {np.sum(test_labels == 1)}"
)

print("\nConfusion Matrix:")
print(conf_matrix)

metrics_lines = [
    "=" * 60,
    "TEST METRICS",
    "=" * 60,
    "Model: BiLSTM (Weighted / Improved)",
    f"Dataset: {DATA_PATH}",
    f"Best Threshold: {best_threshold:.3f}",
    f"Best Validation Macro F1: {best_val_macro_f1:.4f}",
    f"Negative Class Weight: {negative_weight_value:.4f}",
    f"Positive Class Weight: {positive_weight_value:.4f}",
    f"Accuracy: {test_metrics['accuracy']:.4f}",
    f"Precision: {test_metrics['positive_precision']:.4f}",
    f"Recall: {test_metrics['positive_recall']:.4f}",
    f"F1-Score: {test_metrics['positive_f1']:.4f}",
    f"Macro F1: {test_metrics['macro_f1']:.4f}",
    f"Negative Precision: {test_metrics['negative_precision']:.4f}",
    f"Negative Recall: {test_metrics['negative_recall']:.4f}",
    f"Negative F1: {test_metrics['negative_f1']:.4f}",
    f"Negative Support: {np.sum(test_labels == 0)}",
    f"Positive Precision: {test_metrics['positive_precision']:.4f}",
    f"Positive Recall: {test_metrics['positive_recall']:.4f}",
    f"Positive F1: {test_metrics['positive_f1']:.4f}",
    f"Positive Support: {np.sum(test_labels == 1)}",
    "Confusion Matrix:",
    str(conf_matrix),
]
save_text_report(TEST_METRICS_PATH, metrics_lines)
print(f"Detailed test metrics saved to: {TEST_METRICS_PATH}")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker="o")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="o")
plt.title("Weighted Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(val_macro_f1_scores) + 1), val_macro_f1_scores, label="Val Macro F1", marker="o")
plt.plot(range(1, len(val_negative_recalls) + 1), val_negative_recalls, label="Val Neg Recall", marker="o")
plt.title("Validation Macro F1 and Neg Recall")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(LOSS_CURVES_PATH, dpi=150)
plt.close()
print(f"\nLoss curves saved to: {LOSS_CURVES_PATH}")

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix,
    display_labels=["Negative", "Positive"],
)
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Weighted / Improved BiLSTM - Confusion Matrix")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
plt.close()
print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")

model_size_mb = sum(param.numel() * param.element_size() for param in model.parameters()) / (1024 * 1024)
sample_inputs, _ = next(iter(test_loader))
sample_inputs = sample_inputs.to(device)

with torch.no_grad():
    _ = model(sample_inputs)

inference_times = []
with torch.no_grad():
    for _ in range(10):
        start = time.time()
        _ = model(sample_inputs)
        inference_times.append(time.time() - start)

avg_inference_time = np.mean(inference_times)

summary_lines = [
    "=" * 60,
    "MODEL SUMMARY",
    "=" * 60,
    "  Model:             BiLSTM (Weighted / Improved)",
    f"  Total Parameters:  {sum(param.numel() for param in model.parameters()):,}",
    f"  Model Size:        {model_size_mb:.1f} MB",
    f"  Training Time:     {total_training_time:.1f}s ({total_training_time / 60:.1f} min)",
    f"  Avg Inference Time (batch of {BATCH_SIZE}): {avg_inference_time * 1000:.1f} ms",
    f"  Best Threshold:    {best_threshold:.3f}",
    f"  Test Accuracy:     {test_metrics['accuracy']:.4f}",
    f"  Test F1-Score:     {test_metrics['positive_f1']:.4f}",
    f"  Macro F1:          {test_metrics['macro_f1']:.4f}",
    f"  Neg Precision:     {test_metrics['negative_precision']:.4f}",
    f"  Neg Recall:        {test_metrics['negative_recall']:.4f}",
]
summary_text = save_text_report(SUMMARY_PATH, summary_lines)
print("\n" + summary_text, end="")
print(f"Model summary saved to: {SUMMARY_PATH}")

baseline_metrics = parse_metrics_report(BASELINE_METRICS_PATH)
if baseline_metrics:
    baseline_negative_precision = float(baseline_metrics["Negative Precision"])
    baseline_negative_recall = float(baseline_metrics["Negative Recall"])
    baseline_positive_f1 = float(baseline_metrics["F1-Score"])

    comparison_lines = [
        "=" * 60,
        "COMPARISON TO UNWEIGHTED",
        "=" * 60,
        f"Unweighted Negative Precision: {baseline_negative_precision:.4f}",
        f"Improved Weighted Negative Precision: {test_metrics['negative_precision']:.4f}",
        f"Precision Improvement: {test_metrics['negative_precision'] - baseline_negative_precision:+.4f}",
        f"Unweighted Negative Recall: {baseline_negative_recall:.4f}",
        f"Improved Weighted Negative Recall: {test_metrics['negative_recall']:.4f}",
        f"Recall Improvement: {test_metrics['negative_recall'] - baseline_negative_recall:+.4f}",
        f"Unweighted Positive F1: {baseline_positive_f1:.4f}",
        f"Improved Weighted Positive F1: {test_metrics['positive_f1']:.4f}",
        f"F1 Difference: {test_metrics['positive_f1'] - baseline_positive_f1:+.4f}",
    ]
    comparison_text = save_text_report(COMPARISON_PATH, comparison_lines)
    print("\n" + comparison_text, end="")
    print(f"Comparison report saved to: {COMPARISON_PATH}")

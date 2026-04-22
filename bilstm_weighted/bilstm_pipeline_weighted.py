"""
CS6120 Final Project - BiLSTM Pipeline (Weighted)
Author: Yu Ye & Xuyang Shen
"""

# --- CELL 1: Setup and Downloads ---
# In Colab, run this to download the dataset and GloVe embeddings
# !wget https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Software_5.json.gz
# !wget http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
# !unzip -q glove.6B.zip -d glove_data

# --- CELL 2: Imports ---
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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

# Setting random seeds for reproducibility
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
    """Search multiple candidate directories for a required data file or folder."""
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


DATA_PATH = resolve_resource("Software_5.json.gz")
GLOVE_PATH = resolve_resource("glove_data", expect_dir=True) / "glove.6B.300d.txt"
MODEL_PATH = OUTPUT_DIR / "model.pt"
LOSS_CURVES_PATH = OUTPUT_DIR / "loss_curves.png"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
TEST_METRICS_PATH = OUTPUT_DIR / "test_metrics.txt"
SUMMARY_PATH = OUTPUT_DIR / "model_summary.txt"


# SECTION 1: DATA LOADING AND PREPROCESSING


# --- CELL 3: Data Loading and Preprocessing ---
def load_amazon_data(file_path, max_reviews=100000):
    """Load up to max_reviews records from a gzipped JSON lines file."""
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= max_reviews:
                break
            record = json.loads(line)
            if 'reviewText' in record and 'overall' in record:
                data.append({
                    'text': record['reviewText'],
                    'rating': float(record['overall'])
                })
    return pd.DataFrame(data)

# Load data
print(f"Artifacts will be saved to: {OUTPUT_DIR}")
print(f"Using dataset: {DATA_PATH}")
print(f"Using GloVe embeddings: {GLOVE_PATH}")
df = load_amazon_data(DATA_PATH, max_reviews=100000)

# Clean and convert labels
df = df[df['rating'] != 3.0]
df['label'] = df['rating'].apply(lambda x: 1 if x > 3.0 else 0)

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens

df['tokens'] = df['text'].apply(clean_text)

# Train/Val/Test Split = 80/10/10
train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
train_df, val_df = train_test_split(train_val_df, test_size=1/9, random_state=42, stratify=train_val_df['label'])

print(f"  Total reviews (after excluding 3-star): {len(df)}")
print(f"  Train size: {len(train_df)}")
print(f"  Val size:   {len(val_df)}")
print(f"  Test size:  {len(test_df)}")
print(f"  Positive ratio: {df['label'].mean():.2%}")

# Compute class weights from training set
class_counts = np.bincount(train_df['label'].astype(int))
negative_count = int(class_counts[0])
positive_count = int(class_counts[1])
negative_weight_value = len(train_df) / (2 * negative_count)
positive_weight_value = len(train_df) / (2 * positive_count)

print(f"  Negative samples (train): {negative_count}")
print(f"  Positive samples (train): {positive_count}")
print(f"  Negative class weight: {negative_weight_value:.4f}")
print(f"  Positive class weight: {positive_weight_value:.4f}")


# SECTION 2: VOCABULARY AND EMBEDDINGS


# --- CELL 4: Vocabulary and GloVe Embeddings ---
def build_vocab(sentences, min_freq=2):
    """Build word-to-index vocabulary; words below min_freq are excluded."""
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
print(f"Vocabulary size: {len(vocab)}")

def load_glove_embeddings(glove_path, expected_dim=300):
    """Parse the GloVe text file into a word-to-vector dictionary."""
    embeddings_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            if len(vector) == expected_dim:
                embeddings_dict[word] = vector
    return embeddings_dict

# Load GloVe 300d embeddings
glove_dict = load_glove_embeddings(GLOVE_PATH)

def create_embedding_matrix(vocab, embeddings_dict, embedding_dim=300):
    """Map vocabulary words to GloVe vectors; OOV words get random normal init."""
    matrix = np.zeros((len(vocab), embedding_dim), dtype=np.float32)
    for word, i in vocab.items():
        if word in embeddings_dict:
            matrix[i] = embeddings_dict[word]
        else:
            # Random init for words not in GloVe; pad/unk stay as zero vectors
            if word not in ["<pad>", "<unk>"]:
                matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,)).astype(np.float32)
    return matrix

embedding_matrix = create_embedding_matrix(vocab, glove_dict)
print(f"Embedding matrix shape: {embedding_matrix.shape}")

# --- CELL 5: PyTorch Dataset Preparation ---
MAX_SEQ_LENGTH = 256

def tokens_to_indices(tokens, vocab, max_len):
    """Convert a token list to a fixed-length index sequence"""
    indices = [vocab.get(w, vocab["<unk>"]) for w in tokens]
    if len(indices) < max_len:
        indices += [vocab["<pad>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices

class AmazonReviewDataset(Dataset):
    """PyTorch Dataset wrapping tokenized Amazon reviews and binary sentiment labels."""

    def __init__(self, df, vocab, max_len):
        self.labels = df['label'].astype(np.float32).values
        # Pre-convert token lists to index sequences at construction time for speed
        self.sequences = df['tokens'].apply(lambda t: tokens_to_indices(t, vocab, max_len)).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

train_dataset = AmazonReviewDataset(train_df, vocab, MAX_SEQ_LENGTH)
val_dataset = AmazonReviewDataset(val_df, vocab, MAX_SEQ_LENGTH)
test_dataset = AmazonReviewDataset(test_df, vocab, MAX_SEQ_LENGTH)

BATCH_SIZE = 64

# WeightedRandomSampler draws batches with approximately equal class counts
sample_weights = np.where(train_df['label'].values == 0, 1.0 / negative_count, 1.0 / positive_count)
balanced_sampler = WeightedRandomSampler(
    weights=torch.as_tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=balanced_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nDataLoaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches:   {len(val_loader)}")
print(f"  Test batches:  {len(test_loader)}")


# SECTION 3: MODEL SETUP


# --- CELL 6: Model Architecture ---
class BiLSTMWeightedSentiment(nn.Module):
    """Bidirectional LSTM sentiment classifier with max pooling over time steps."""

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


def calculate_metrics(labels, predictions):
    """Compute accuracy, macro precision/recall/F1, and confusion matrix."""
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=None, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_precision": (prec[0] + prec[1]) / 2,
        "macro_recall": (rec[0] + rec[1]) / 2,
        "macro_f1": (f1[0] + f1[1]) / 2,
        "negative_recall": rec[0],  # used for threshold tiebreak only
        "confusion_matrix": confusion_matrix(labels, predictions),
    }


def collect_probabilities(model, loader, device):
    """Run inference over a DataLoader and return sigmoid probabilities and true labels."""
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
    """Compute weighted validation loss."""
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            batch_loss = criterion(logits, labels)
            batch_weights = torch.where(labels < 0.5, negative_weight, positive_weight)
            total_loss += (batch_loss * batch_weights).mean().item()
    return total_loss / max(1, len(loader))


def search_best_threshold(probabilities, labels):
    """Grid search over 33 thresholds (0.10-0.90) to maximize macro F1; tiebreak on negative recall."""
    best_result = None
    for threshold in np.linspace(0.10, 0.90, 33):
        predictions = (probabilities >= threshold).astype(int)
        metrics = calculate_metrics(labels, predictions)
        result = {"threshold": float(threshold), "metrics": metrics}

        if best_result is None:
            best_result = result
            continue

        current_f1 = metrics["macro_f1"]
        best_f1 = best_result["metrics"]["macro_f1"]
        if current_f1 > best_f1:
            best_result = result
        elif np.isclose(current_f1, best_f1):
            if metrics["negative_recall"] > best_result["metrics"]["negative_recall"]:
                best_result = result

    return best_result


HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT_PROB = 0.5

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"\nUsing device: {device}")

model = BiLSTMWeightedSentiment(
    len(vocab), 300, HIDDEN_DIM, embedding_matrix,
    num_layers=NUM_LAYERS, dropout_prob=DROPOUT_PROB,
).to(device)

print(f"\nModel: BiLSTM (Weighted)")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# SECTION 4: TRAINING LOOP


# --- CELL 7: Training Loop with Early Stopping ---
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=8e-4)
criterion = nn.BCEWithLogitsLoss(reduction='none')
negative_weight = torch.tensor(negative_weight_value, dtype=torch.float32, device=device)
positive_weight = torch.tensor(positive_weight_value, dtype=torch.float32, device=device)

EPOCHS = 10
PATIENCE = 4

best_val_macro_f1 = float('-inf')
best_threshold = 0.5
patience_counter = 0

train_losses = []
val_losses = []
val_macro_f1_scores = []
training_start_time = time.time()

for epoch in range(EPOCHS):
    # Training Loop
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        batch_loss = criterion(logits, labels)
        batch_weights = torch.where(labels < 0.5, negative_weight, positive_weight)
        loss = (batch_loss * batch_weights).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation Loop
    model.eval()
    avg_val_loss = evaluate_loss(model, val_loader, criterion, device, negative_weight, positive_weight)
    val_losses.append(avg_val_loss)

    val_probabilities, val_labels = collect_probabilities(model, val_loader, device)
    threshold_result = search_best_threshold(val_probabilities, val_labels)
    epoch_threshold = threshold_result["threshold"]
    epoch_metrics = threshold_result["metrics"]
    val_macro_f1_scores.append(epoch_metrics["macro_f1"])

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Macro F1: {epoch_metrics['macro_f1']:.4f} - Threshold: {epoch_threshold:.3f}")

    # Early Stopping Logic
    if epoch_metrics["macro_f1"] > best_val_macro_f1:
        best_val_macro_f1 = epoch_metrics["macro_f1"]
        best_threshold = epoch_threshold
        patience_counter = 0
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "best_threshold": best_threshold,
            "best_val_macro_f1": best_val_macro_f1,
            "vocab": vocab,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "embedding_dim": 300,
            "dropout_prob": DROPOUT_PROB,
            "negative_weight": negative_weight_value,
            "positive_weight": positive_weight_value,
        }
        torch.save(checkpoint, MODEL_PATH)
        print(f"  Validation macro F1 improved. Model saved.")
    else:
        patience_counter += 1
        print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

total_training_time = time.time() - training_start_time
print(f"\nTraining complete. Total time: {total_training_time:.1f}s ({total_training_time/60:.1f} min)")


# SECTION 5: EVALUATION AND VISUALIZATION


# --- CELL 8: Final Evaluation on Test Set ---
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
best_threshold = float(checkpoint["best_threshold"])
best_val_macro_f1 = float(checkpoint["best_val_macro_f1"])
model.eval()

test_probabilities, test_labels = collect_probabilities(model, test_loader, device)
test_preds_binary = (test_probabilities >= best_threshold).astype(int)
test_metrics = calculate_metrics(test_labels, test_preds_binary)
conf_matrix = test_metrics["confusion_matrix"]

print(f"\nBest threshold: {best_threshold:.3f}  (from validation)")
print(f"\nTest Accuracy:  {test_metrics['accuracy']:.4f}")
print(f"Test Precision: {test_metrics['macro_precision']:.4f}")
print(f"Test Recall:    {test_metrics['macro_recall']:.4f}")
print(f"Test F1-Score:  {test_metrics['macro_f1']:.4f}")

# Confusion Matrix
print(f"\nConfusion Matrix:")
print(conf_matrix)

metrics_lines = [
    "Model: BiLSTM (Weighted)",
    f"Dataset: {DATA_PATH}",
    f"Best Threshold: {best_threshold:.3f}",
    f"Negative Class Weight: {negative_weight_value:.4f}",
    f"Positive Class Weight: {positive_weight_value:.4f}",
    f"Accuracy: {test_metrics['accuracy']:.4f}",
    f"Precision: {test_metrics['macro_precision']:.4f}",
    f"Recall: {test_metrics['macro_recall']:.4f}",
    f"F1-Score: {test_metrics['macro_f1']:.4f}",
    "Confusion Matrix:",
    str(conf_matrix),
]
save_text_report(TEST_METRICS_PATH, metrics_lines)
print(f"Detailed test metrics saved to: {TEST_METRICS_PATH}")

# --- CELL 9: Visualizations ---
# 1. Training vs Validation Loss Curves + Val Macro F1
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Validation Macro F1
plt.subplot(1, 2, 2)
plt.plot(range(1, len(val_macro_f1_scores) + 1), val_macro_f1_scores, label='Val Macro F1',
         marker='o', color='green')
plt.title('Validation Macro F1 over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Macro F1')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(LOSS_CURVES_PATH, dpi=150)
plt.close()
print(f"\nLoss curves saved to: {LOSS_CURVES_PATH}")

# 3. Confusion Matrix Visualization
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix,
    display_labels=['Negative', 'Positive']
)
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('BiLSTM (Weighted) - Confusion Matrix on Test Set')
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
plt.close()
print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")

# --- CELL 10: Model Summary for Comparison ---
# Calculate model size
model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

# Measure inference speed
model.eval()
sample_inputs, sample_labels = next(iter(test_loader))
sample_inputs = sample_inputs.to(device)

# Warm up
with torch.no_grad():
    _ = model(sample_inputs)

# Time inference
inference_times = []
with torch.no_grad():
    for _ in range(10):
        start = time.time()
        _ = model(sample_inputs)
        inference_times.append(time.time() - start)

avg_inference_time = np.mean(inference_times)

summary_lines = [
    "MODEL SUMMARY",
    "  Model:             BiLSTM (Weighted)",
    f"  Best Threshold:    {best_threshold:.3f}",
    f"  Total Parameters:  {sum(p.numel() for p in model.parameters()):,}",
    f"  Model Size:        {model_size_mb:.1f} MB",
    f"  Training Time:     {total_training_time:.1f}s ({total_training_time/60:.1f} min)",
    f"  Avg Inference Time (batch of {BATCH_SIZE}): {avg_inference_time*1000:.1f} ms",
    f"  Test Accuracy:     {test_metrics['accuracy']:.4f}",
    f"  Test F1-Score:     {test_metrics['macro_f1']:.4f}",
]
summary_text = save_text_report(SUMMARY_PATH, summary_lines)
print("\n" + summary_text, end="")
print(f"Model summary saved to: {SUMMARY_PATH}")

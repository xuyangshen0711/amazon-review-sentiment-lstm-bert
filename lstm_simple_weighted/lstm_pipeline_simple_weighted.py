"""
CS6120 Final Project - LSTM Pipeline (Simple Weighted Loss)
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
    """
    Loads a subset of Amazon reviews from the gzipped JSON lines file.
    """
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

# Compute class weight from training set
num_pos = int(train_df['label'].sum())
num_neg = int(len(train_df) - num_pos)
pos_weight_value = num_neg / num_pos
print(f"  pos_weight: {pos_weight_value:.4f}  ({num_neg}/{num_pos})")


# SECTION 2: VOCABULARY AND EMBEDDINGS


# --- CELL 4: Vocabulary and GloVe Embeddings ---
def build_vocab(sentences, min_freq=2):
    """Build word-to-index vocabulary from training tokens.

    Words below min_freq are excluded to reduce noise.
    Index 0 = <pad> (padding), index 1 = <unk> (unknown words).
    """
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
    matrix = np.zeros((len(vocab), embedding_dim))
    for word, i in vocab.items():
        if word in embeddings_dict:
            matrix[i] = embeddings_dict[word]
        else:
            # Random init for words not in GloVe; pad/unk stay as zero vectors
            if word not in ["<pad>", "<unk>"]:
                matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
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
        self.labels = df['label'].values
        # Pre-convert token lists to index sequences at construction time for speed
        self.sequences = df['tokens'].apply(lambda t: tokens_to_indices(t, vocab, max_len)).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

train_dataset = AmazonReviewDataset(train_df, vocab, MAX_SEQ_LENGTH)
val_dataset = AmazonReviewDataset(val_df, vocab, MAX_SEQ_LENGTH)
test_dataset = AmazonReviewDataset(test_df, vocab, MAX_SEQ_LENGTH)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nDataLoaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches:   {len(val_loader)}")
print(f"  Test batches:  {len(test_loader)}")


# SECTION 3: MODEL SETUP


# --- CELL 6: Model Architecture ---
class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, embedding_matrix, num_layers=2, dropout_prob=0.5):
        super(LSTMSentiment, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Use final hidden state of the last layer
        final_hidden = hidden[-1, :, :]

        out = self.dropout(final_hidden)
        out = self.fc(out)
        return out.squeeze()

# Instantiate model
HIDDEN_DIM = 256
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"\nUsing device: {device}")
model = LSTMSentiment(len(vocab), 300, HIDDEN_DIM, embedding_matrix).to(device)

print(f"\nModel: LSTM (Simple Weighted Loss)")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# SECTION 4: TRAINING LOOP


# --- CELL 7: Training Loop with Early Stopping ---
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

EPOCHS = 10
PATIENCE = 3

best_val_loss = float('inf')
patience_counter = 0

train_losses = []
val_losses = []
val_accuracies = []
training_start_time = time.time()

for epoch in range(EPOCHS):
    # Training Loop
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation Loop
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = model(inputs)
            loss = criterion(predictions, labels)
            val_loss += loss.item()

            preds = (predictions > 0).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_acc = accuracy_score(all_labels, all_preds)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f}")

    # Early Stopping Logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save({
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": 2,
            "embedding_dim": 300,
            "dropout_prob": 0.5,
            "best_threshold": 0.5,
        }, MODEL_PATH)
        print(f"  Validation loss improved. Model saved.")
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
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False)["model_state_dict"])
model.eval()

test_preds, test_targets = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        predictions = model(inputs)

        test_preds.extend(predictions.cpu().numpy())
        test_targets.extend(labels.cpu().numpy())

# Convert logits to binary predictions with threshold 0
test_preds_binary = [1 if p > 0 else 0 for p in test_preds]

acc = accuracy_score(test_targets, test_preds_binary)
precision, recall, f1, _ = precision_recall_fscore_support(test_targets, test_preds_binary, average='macro')

print(f"\nTest Accuracy:  {acc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall:    {recall:.4f}")
print(f"Test F1-Score:  {f1:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(test_targets, test_preds_binary)
print(f"\nConfusion Matrix:")
print(conf_matrix)

metrics_lines = [
    "Model: LSTM (Simple Weighted Loss)",
    f"Dataset: {DATA_PATH}",
    f"pos_weight: {pos_weight_value:.4f}",
    f"Accuracy: {acc:.4f}",
    f"Precision: {precision:.4f}",
    f"Recall: {recall:.4f}",
    f"F1-Score: {f1:.4f}",
    "Confusion Matrix:",
    str(conf_matrix),
]
save_text_report(TEST_METRICS_PATH, metrics_lines)
print(f"Detailed test metrics saved to: {TEST_METRICS_PATH}")

# --- CELL 9: Visualizations ---
# 1. Training vs Validation Loss Curves + Validation Accuracy
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Validation Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Accuracy',
         marker='o', color='green')
plt.title('Validation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
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
plt.title('LSTM (Simple Weighted) - Confusion Matrix on Test Set')
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
    "  Model:             LSTM (Simple Weighted Loss)",
    f"  pos_weight:        {pos_weight_value:.4f}",
    f"  Total Parameters:  {sum(p.numel() for p in model.parameters()):,}",
    f"  Model Size:        {model_size_mb:.1f} MB",
    f"  Training Time:     {total_training_time:.1f}s ({total_training_time/60:.1f} min)",
    f"  Avg Inference Time (batch of {BATCH_SIZE}): {avg_inference_time*1000:.1f} ms",
    f"  Test Accuracy:     {acc:.4f}",
    f"  Test F1-Score:     {f1:.4f}",
]
summary_text = save_text_report(SUMMARY_PATH, summary_lines)
print("\n" + summary_text, end="")
print(f"Model summary saved to: {SUMMARY_PATH}")

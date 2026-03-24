"""
CS6120 Final Project - BERT Pipeline
Author: Yu Ye

==============================================================
INSTALLATION
==============================================================
Install required dependencies with:
    pip install torch transformers numpy pandas scikit-learn matplotlib

==============================================================
HOW TO RUN (Local)
==============================================================
1. Place Software_5.json.gz in the same directory as this script.
2. Run the full pipeline:
       python bert_pipeline.py
   Output: best_bert_model.pt, bert_loss_curves.png, bert_confusion_matrix.png
   (Results are printed to stdout; redirect to save: python bert_pipeline.py > bert_result.txt)

==============================================================
HOW TO RUN (Google Colab)
==============================================================
You can copy each section delineated by "--- CELL ---" into a
separate Colab cell. Run Cell 1 first to install dependencies
and download the dataset.
==============================================================
"""

# --- CELL 1: Setup and Downloads (Colab Only) ---
# In Colab, run these commands first:
# !pip install torch transformers numpy pandas scikit-learn matplotlib
# !wget --no-check-certificate https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Software_5.json.gz

# --- CELL 2: Imports ---
import os
import gzip
import json
import time
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
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

# Setting random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# SECTION 1: DATA LOADING AND PREPROCESSING


# --- CELL 3: Data Loading ---
def load_amazon_data(file_path, max_reviews=100000):
    """
    Loads a subset of Amazon reviews from the gzipped JSON lines file.
    This function is identical to the LSTM pipeline to ensure
    both models train on the exact same data.
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
df = load_amazon_data('Software_5.json.gz', max_reviews=100000)

# Clean and convert labels
# 4-5 stars -> 1 (Positive), 1-2 stars -> 0 (Negative), 3-star reviews excluded
df = df[df['rating'] != 3.0]
df['label'] = df['rating'].apply(lambda x: 1 if x > 3.0 else 0)

# Train/Val/Test Split = 80/10/10
train_val_df, test_df = train_test_split(
    df, test_size=0.1, random_state=42, stratify=df['label']
)
train_df, val_df = train_test_split(
    train_val_df, test_size=1/9, random_state=42, stratify=train_val_df['label']
)

print(f"Dataset loaded successfully!")
print(f"  Total reviews (after excluding 3-star): {len(df)}")
print(f"  Train size: {len(train_df)}")
print(f"  Val size:   {len(val_df)}")
print(f"  Test size:  {len(test_df)}")
print(f"  Positive ratio: {df['label'].mean():.2%}")


# SECTION 2: BERT TOKENIZATION

# --- CELL 4: BERT Tokenizer ---
MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 256  

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class AmazonReviewDataset(Dataset):
    """
    PyTorch Dataset for Amazon reviews with BERT tokenization.
    """
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,    # Add [CLS] and [SEP] tokens
            max_length=self.max_length,
            padding='max_length',        # Pad to max_length
            truncation=True,             # Truncate if longer than max_length
            return_tensors='pt'          # Return PyTorch tensors
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),        # Shape: (max_length,)
            'attention_mask': encoding['attention_mask'].squeeze(0),  # Shape: (max_length,)
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# --- CELL 5: Create DataLoaders ---
BATCH_SIZE = 16  # Smaller batch size than LSTM due to BERT's memory requirements

train_dataset = AmazonReviewDataset(train_df['text'], train_df['label'], tokenizer, MAX_LENGTH)
val_dataset = AmazonReviewDataset(val_df['text'], val_df['label'], tokenizer, MAX_LENGTH)
test_dataset = AmazonReviewDataset(test_df['text'], test_df['label'], tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nDataLoaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches:   {len(val_loader)}")
print(f"  Test batches:  {len(test_loader)}")


# SECTION 3: MODEL SETUP


# --- CELL 6: Model, Optimizer, Scheduler ---
# Device selection
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"\nUsing device: {device}")

# Load pretrained BERT with a classification head (2 classes: Negative, Positive)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)
model.to(device)

# Hyperparameters for fine-tuning
LEARNING_RATE = 2e-5   # Standard learning rate for BERT fine-tuning
EPOCHS = 4             # BERT typically converges in 2-4 epochs
WARMUP_RATIO = 0.1     # Warm up for 10% of total training steps

# AdamW optimizer (handles weight decay properly for BERT)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# Linear learning rate scheduler with warmup
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"\nModel: {MODEL_NAME}")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Total training steps: {total_steps}")
print(f"  Warmup steps: {warmup_steps}")


# SECTION 4: TRAINING LOOP


# --- CELL 7: Training ---
def train_one_epoch(model, data_loader, optimizer, scheduler, device):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"    Batch {batch_idx + 1}/{len(data_loader)} - Loss: {loss.item():.4f}")

    return total_loss / len(data_loader)


def evaluate(model, data_loader, device):
    """Evaluate model and return average loss and predictions."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()

            # Get predictions (argmax of logits)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, all_preds, all_labels


# Main training loop

train_losses = []
val_losses = []
val_accuracies = []
best_val_loss = float('inf')
training_start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 40)

    # Train
    avg_train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
    train_losses.append(avg_train_loss)

    # Validate
    avg_val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    epoch_time = time.time() - epoch_start_time
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss:   {avg_val_loss:.4f}")
    print(f"  Val Acc:    {val_acc:.4f}")
    print(f"  Time:       {epoch_time:.1f}s")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_bert_model.pt')
        print(f"  -> Validation loss improved. Model saved.")

total_training_time = time.time() - training_start_time
print(f"\nTraining complete! Total time: {total_training_time:.1f}s ({total_training_time/60:.1f} min)")


# SECTION 5: EVALUATION AND VISUALIZATION


# --- CELL 8: Final Evaluation on Test Set ---
# Load the best model weights
model.load_state_dict(torch.load('best_bert_model.pt'))


test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, device)

# Calculate detailed metrics
precision, recall, f1, _ = precision_recall_fscore_support(
    test_labels, test_preds, average='binary'
)

print(f"\nTest Accuracy:  {test_acc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall:    {recall:.4f}")
print(f"Test F1-Score:  {f1:.4f}")
print(f"Test Loss:      {test_loss:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(test_labels, test_preds)
print(f"\nConfusion Matrix:")
print(conf_matrix)

# --- CELL 9: Visualizations ---
# 1. Training vs Validation Loss Curves
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
plt.savefig('bert_loss_curves.png', dpi=150)
plt.close()
print("\nLoss curves saved to: bert_loss_curves.png")

# 3. Confusion Matrix Visualization
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix,
    display_labels=['Negative', 'Positive']
)
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('BERT - Confusion Matrix on Test Set')
plt.tight_layout()
plt.savefig('bert_confusion_matrix.png', dpi=150)
plt.close()
print("Confusion matrix saved to: bert_confusion_matrix.png")

# --- CELL 10: Model Summary for Comparison ---
print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)

# Calculate model size
model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

# Measure inference speed on a small sample
model.eval()
sample_batch = next(iter(test_loader))
sample_input_ids = sample_batch['input_ids'].to(device)
sample_attention_mask = sample_batch['attention_mask'].to(device)

# Warm up
with torch.no_grad():
    _ = model(input_ids=sample_input_ids, attention_mask=sample_attention_mask)

# Time inference
inference_times = []
with torch.no_grad():
    for _ in range(10):
        start = time.time()
        _ = model(input_ids=sample_input_ids, attention_mask=sample_attention_mask)
        inference_times.append(time.time() - start)

avg_inference_time = np.mean(inference_times)

print(f"  Model:             {MODEL_NAME}")
print(f"  Total Parameters:  {sum(p.numel() for p in model.parameters()):,}")
print(f"  Model Size:        {model_size_mb:.1f} MB")
print(f"  Training Time:     {total_training_time:.1f}s ({total_training_time/60:.1f} min)")
print(f"  Avg Inference Time (batch of {BATCH_SIZE}): {avg_inference_time*1000:.1f} ms")
print(f"  Test Accuracy:     {test_acc:.4f}")
print(f"  Test F1-Score:     {f1:.4f}")

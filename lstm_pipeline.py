"""
CS6120 Final Project - LSTM Pipeline
Author: Xuyang Shen

==============================================================
INSTALLATION
==============================================================
Install required dependencies with:
    pip install torch numpy pandas scikit-learn matplotlib nltk

Then download NLTK stopwords (run once):
    python -c "import nltk; nltk.download('stopwords')"

==============================================================
HOW TO RUN (Local)
==============================================================
1. Place Software_5.json.gz in the same directory as this script.
2. Unzip GloVe embeddings into a subfolder named glove_data/:
       unzip glove.6B.zip -d glove_data
3. Run the full pipeline:
       python lstm_pipeline.py
   Output: best_lstm_model.pt, loss_curves.png
   (Results are printed to stdout; redirect to save: python lstm_pipeline.py > result.txt)

==============================================================
HOW TO RUN (Google Colab)
==============================================================
You can copy each section delineated by "--- CELL ---" into a
separate Colab cell. Run Cell 1 first to download the dataset
and GloVe embeddings.
==============================================================
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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Setting random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

STOP_WORDS = set(stopwords.words('english'))

# --- CELL 3: Data Loading and Preprocessing ---
def load_amazon_data(file_path, max_reviews=100000):
    """Loads a subset of Amazon reviews from the gzipped JSON lines file."""
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

# Please run the bash commands in Cell 1 first to get the Software_5.json.gz dataset
# You can replace this path if you are using a different category subset.
df = load_amazon_data('Software_5.json.gz', max_reviews=100000)

# Clean and convert labels
# 4-5 stars -> 1 (Positive), 1-2 stars -> 0 (Negative), 3-star reviews excluded
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

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

# --- CELL 4: Vocabulary and GloVe Embeddings ---
def build_vocab(sentences, min_freq=2):
    counter = Counter()
    for tokens in sentences:
        counter.update(tokens)
    
    # 0 = <pad>, 1 = <unk>
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
    embeddings_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            if len(vector) == expected_dim:
                embeddings_dict[word] = vector
    return embeddings_dict

# Load GloVe 300d embeddings (this file format assumes the 6B corpus)
glove_dict = load_glove_embeddings('glove_data/glove.6B.300d.txt')

def create_embedding_matrix(vocab, embeddings_dict, embedding_dim=300):
    matrix = np.zeros((len(vocab), embedding_dim))
    for word, i in vocab.items():
        if word in embeddings_dict:
            matrix[i] = embeddings_dict[word]
        else:
            if word not in ["<pad>", "<unk>"]:
                # Initialize random vector for out-of-vocabulary words
                matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return matrix

embedding_matrix = create_embedding_matrix(vocab, glove_dict)
print(f"Embedding matrix shape: {embedding_matrix.shape}")

# --- CELL 5: PyTorch Dataset Preparation ---
MAX_SEQ_LENGTH = 256 # Determined via EDA similarity to BERT part

def tokens_to_indices(tokens, vocab, max_len):
    indices = [vocab.get(w, vocab["<unk>"]) for w in tokens]
    if len(indices) < max_len:
        indices += [vocab["<pad>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices

class AmazonReviewDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.labels = df['label'].values
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

# --- CELL 6: Model Architecture ---
class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, embedding_matrix, num_layers=2, dropout_prob=0.5):
        super(LSTMSentiment, self).__init__()
        
        # 1. Embedding Layer (Frozen)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False # Freeze weights
        
        # 2. 2-layer LSTM
        # We set batch_first=True so inputs are (batch, seq, feature)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        
        # 3. Dropout
        self.dropout = nn.Dropout(dropout_prob)
        
        # 4. Fully Connected Layer -> outputs continuous value 
        self.fc = nn.Linear(hidden_dim, 1)
        # Sigmoid for binary classification
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use final hidden state of the last layer to represent the entire sequence
        final_hidden = hidden[-1, :, :]
        
        out = self.dropout(final_hidden)
        out = self.fc(out)
        return self.sigmoid(out).squeeze()

# Instantiate model
HIDDEN_DIM = 256
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")
model = LSTMSentiment(len(vocab), 300, HIDDEN_DIM, embedding_matrix).to(device)

# --- CELL 7: Training Loop with Early Stopping ---
# Ensure we only pass parameters that require gradients to the optimizer
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
criterion = nn.BCELoss() # Binary Cross Entropy Loss

EPOCHS = 10
PATIENCE = 3

best_val_loss = float('inf')
patience_counter = 0

train_losses = []
val_losses = []

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
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = model(inputs)
            loss = criterion(predictions, labels)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
    
    # Early Stopping Logic (Monitoring Validation Loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save the best model state
        torch.save(model.state_dict(), 'best_lstm_model.pt')
        print(f"  -> Validation loss improved. Model saved.")
    else:
        patience_counter += 1
        print(f"  -> No improvement. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered!")
            break

# --- CELL 8: Evaluation & Visualization ---
# Load the best model weights
model.load_state_dict(torch.load('best_lstm_model.pt'))
model.eval()

# Plot Training vs Validation Loss Curves
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curves.png')
plt.close()

# Run Evaluation on Test Set
test_preds, test_targets = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        predictions = model(inputs)
        
        test_preds.extend(predictions.cpu().numpy())
        test_targets.extend(labels.cpu().numpy())

# Convert probabilities to binary predictions with 0.5 threshold
test_preds_binary = [1 if p >= 0.5 else 0 for p in test_preds]

# Calculate Metrics
acc = accuracy_score(test_targets, test_preds_binary)
precision, recall, f1, _ = precision_recall_fscore_support(test_targets, test_preds_binary, average='binary')

print(f"Test Accuracy:  {acc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall:    {recall:.4f}")
print(f"Test F1-Score:  {f1:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(test_targets, test_preds_binary)
print("Confusion Matrix:")
print(conf_matrix)

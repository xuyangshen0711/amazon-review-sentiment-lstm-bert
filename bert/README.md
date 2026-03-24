# BERT Sentiment Analysis Pipeline

This directory contains the BERT-based implementation for the CS6120 Final Project (Team 102). It serves as a high-performance comparison model against the baseline LSTM architecture for Amazon Software review sentiment analysis.

## Overview

Unlike the LSTM model which relies on static GloVe embeddings and suffers from majority-class collapse due to dataset imbalance (80.2% positive reviews), this BERT pipeline leverages dynamic self-attention to successfully capture both positive and negative sentiments without requiring artificial loss weighting.

## Key Results (Test Set)

Our evaluation on the 10% held-out test set (1,121 reviews) yielded the following metrics:

| Metric | Score |
| --- | --- |
| **Accuracy** | 93.84% |
| **F1-Score** | 0.9622 |
| **Precision** | > 94% |
| **Recall (Positive)** | > 97% |
| **Recall (Negative)** | 78.3% (vs LSTM's 2.25%) |

*Note: The dramatic improvement in Negative Recall demonstrates BERT's ability to natively handle severe class imbalance.*

### Performance Details
- **Hardware:** T4 GPU (Google Colab)
- **Training Time:** ~27.2 minutes (4 Epochs)
- **Model Size:** 417.6 MB
- **Total Parameters:** 109,483,778
- **Inference Time:** ~207.0 ms (per batch of 16)

## Reproducing the Results

Since the `best_bert_model.pt` weights (417MB) are too large for GitHub, you can completely reproduce these exact results by running the deterministic pipeline (`random_state=42`) from scratch.

### 1. Install Dependencies
```bash
pip install torch transformers numpy pandas scikit-learn matplotlib
```

### 2. Download the Dataset
Download the original Amazon reviews subset (`Software_5.json.gz`) into this directory.
```bash
wget --no-check-certificate https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Software_5.json.gz
```

### 3. Run the Pipeline
```bash
python bert_pipeline.py
```

The script will automatically:
1. Load and preprocess the JSON data.
2. Tokenize using `bert-base-uncased`.
3. Train for 4 epochs (saving the best model based on validation loss).
4. Output the evaluating metrics.
5. Generate `bert_loss_curves.png` and `bert_confusion_matrix.png` in this directory.

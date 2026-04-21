# Appendix: User Manual

## 1 README Section

### 1.1 Project Overview

This project implements and compares three LSTM-based sentiment classifiers and one BERT classifier on Amazon Software product reviews. The goal is to classify each review as Positive or Negative. The best-performing model is the **Weighted BiLSTM** (`lstm_weighted/`), which uses a bidirectional LSTM architecture with balanced sampling and class-weighted loss to handle class imbalance.

---

### 1.2 Prerequisites

**Required:**

- **Python:** 3.8 or higher
- **pip:** Python package manager
- **Dataset:** `Software_5.json.gz` (Amazon Software reviews)
- **GloVe Embeddings:** `glove_data/glove.6B.300d.txt` (required for LSTM models only)

**Note:** The BERT pipeline downloads the `bert-base-uncased` tokenizer automatically on first run. No manual download needed.

---

### 1.3 Instructions

#### 1.3.1 Environment Setup

**Step 1: Install Python dependencies**

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install numpy pandas scikit-learn matplotlib nltk
```

**Step 2: Download NLTK stopwords (run once)**

```bash
python3 -c "import nltk; nltk.download('stopwords')"
```

**Step 3: Prepare data files**

Place the following files in the project root (`LSTM/`) or inside any pipeline subfolder — the scripts will find them automatically:

| File | Description |
|------|-------------|
| `Software_5.json.gz` | Amazon Software review dataset |
| `glove_data/glove.6B.300d.txt` | GloVe 300-dimensional word embeddings |

**Expected Project Structure:**

```
LSTM/
├── lstm_unweighted/
│   ├── lstm_pipeline_unweighted.py
│   └── model.pt
├── lstm_weighted/
│   ├── lstm_pipeline_weighted.py
│   ├── demo.py                        ← Interactive demo (best model)
│   └── model.pt
├── lstm_simple_weighted/
│   ├── lstm_pipeline_simple_weighted.py
│   └── model.pt
├── bert_pipeline.py
├── best_bert_model.pt
├── demo.py                            ← Interactive demo (BERT)
└── USER_MANUAL.md
```

---

## 2 Running the Application

### 2.1 Running the Training Pipelines

Each pipeline is fully self-contained. Run from inside its folder:

**Baseline Unweighted BiLSTM**

```bash
cd lstm_unweighted
python3 lstm_pipeline_unweighted.py
```

**Weighted BiLSTM (Best Model)**

```bash
cd lstm_weighted
python3 lstm_pipeline_weighted.py
```

**Simple Weighted LSTM**

```bash
cd lstm_simple_weighted
python3 lstm_pipeline_simple_weighted.py
```

**BERT Fine-tuning**

```bash
python3 bert_pipeline.py
```

Each pipeline automatically saves the following outputs into its own folder:

- `model.pt` — trained model weights (LSTM checkpoints also include vocabulary)
- `loss_curves.png` — training and validation loss curves
- `confusion_matrix.png` — test set confusion matrix
- `test_metrics.txt` — detailed per-class classification metrics
- `model_summary.txt` — model size, inference speed, and performance summary

---

### 2.2 Running the Demo (Recommended)

The demo scripts load a pre-trained model and allow interactive sentiment testing without re-running training.

**BiLSTM Demo (Best Model)**

```bash
cd lstm_weighted
python3 demo.py
```

- **Requirement:** `model.pt` must exist in `lstm_weighted/`

**BERT Demo**

```bash
python3 demo.py
```

- **Requirement:** `best_bert_model.pt` must exist in the project root

Both demos will:

1. Print predictions for 5 built-in sample reviews
2. Enter an interactive mode where you can type your own review
3. Type `quit` (or `exit` / `q`) to exit

---

### 2.3 Expected Output (BiLSTM Demo)

```
============================================================
  BiLSTM Sentiment Analysis — Amazon Reviews Demo
============================================================
  Device : mps
  Model  : Bidirectional LSTM  (weighted / fine-tuned)
  Weights: model.pt
============================================================

Loading model... done.
  Vocab size       : 16,199
  Best threshold   : 0.100

────────────────────────────────────────────────────────────
SAMPLE PREDICTIONS
────────────────────────────────────────────────────────────

[1] This software is absolutely fantastic...
    => Prediction : Positive  (confidence: 99.95%)

[2] Terrible product. It crashed my computer twice...
    => Prediction : Negative  (confidence: 99.99%)

────────────────────────────────────────────────────────────
INTERACTIVE MODE  (type 'quit' to exit)
────────────────────────────────────────────────────────────

Enter a review: Great value for the price!
=> Prediction : Positive  (confidence: 92.10%)
```

---

## 3 Hardware Acceleration

All scripts automatically select the best available compute device. No manual configuration is needed.

| Device | Description |
|--------|-------------|
| **CUDA** | NVIDIA GPU — fastest |
| **MPS** | Apple Silicon (M1/M2/M3) — fast |
| **CPU** | Default fallback — slower training, fully functional |

---

## 4 Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: Software_5.json.gz` | Place the dataset file in the script's folder or project root |
| `FileNotFoundError: glove_data` | Place the `glove_data/` directory in the script's folder or project root |
| `KeyError: 'vocab'` in BiLSTM demo | Re-run `lstm_pipeline_weighted.py` to regenerate `model.pt` |
| `ModuleNotFoundError: transformers` | Run `pip install transformers` |
| BERT download slow on first run | Normal — downloads ~440 MB of pre-trained weights |

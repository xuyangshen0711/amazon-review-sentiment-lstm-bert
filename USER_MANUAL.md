# Appendix

## 1. Project Overview

This project compares LSTM-based sentiment classifiers and a BERT classifier using Amazon product reviews. The following guide walks you through the complete process of training the models on Google Colab and running interactive demos on your local machine.

---

## 2. Data

Before starting, ensure you have the following files available for upload:

- **Dataset**: Software_5.json.gz
- **Embeddings**: glove_data/glove.6B.300d.txt (Required for LSTM models only)

---

## 3. Training on Google Colab

### 3.1 Training LSTM Models (CPU)

1. **Runtime Setup**: Open a new notebook in Google Colab. Go to **Runtime → Change runtime type** and ensure **Hardware accelerator** is set to **None** (CPU).

2. **Upload Files**: Upload the scripts from the `bilstm_weighted/` folder along with the `Software_5.json.gz` dataset and `glove_data/` directory to the Colab environment.

3. **Run Training**: Execute the following commands in a code cell:

```
!pip install torch numpy pandas scikit-learn matplotlib nltk
!python3 lstm_pipeline_weighted.py
```

### 3.2 Training BERT Model (GPU)

1. **Runtime Setup**: Open a new notebook. Go to **Runtime → Change runtime type** and select **T4 GPU** (or any available GPU).

2. **Upload Files**: Upload `bert_pipeline.py` and the `Software_5.json.gz` dataset to the Colab root directory.

3. **Run Training**: Execute the following commands in a code cell:

```
!pip install torch transformers pandas scikit-learn
!python3 bert_pipeline.py
```

---

## 4. Running Local Demos

After training is complete on Google Colab, follow these steps to run the interactive demos on your local machine.

### 4.1 Setup Local Environment

Ensure you have Python 3.8+ installed, then install the necessary libraries:

```
pip install torch transformers numpy pandas scikit-learn matplotlib nltk
```

Also download NLTK stopwords (run once):

```
python3 -c "import nltk; nltk.download('stopwords')"
```

### 4.2 Sync Trained Weights

Download the trained weight files from your Google Colab session to your local project folders:

1. **BiLSTM Weights**: Download `model.pt` and place it inside the `bilstm_weighted/` folder.
2. **BERT Weights**: Download `best_bert_model.pt` and place it in the project root directory.

### 4.3 Execute Demo Scripts

**Step 1: Run BiLSTM Demo (Best Model)**

Navigate to the `bilstm_weighted` folder and launch the demo:

```
cd bilstm_weighted
python3 demo.py
```

**Step 2: Run BERT Demo**

From the project root directory, launch the BERT demo:

```
python3 demo.py
```

**Step 3: Play with the demo**

Once the script starts running, you will first see five sample reviews with predictions. You can then type your own review in the interactive mode and get a Positive / Negative prediction. Type `quit` to exit.

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model configuration
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256
NUM_LABELS = 2
MODEL_PATH = "best_bert_model.pt"

# Sample reviews for inference demo
SAMPLE_REVIEWS = [
    "This software is amazing, it solved all my problems!",
    "Terrible product, crashed every time I tried to use it.",
    "Works as expected, nothing special but gets the job done.",
    "Complete waste of money, the interface is confusing and buggy.",
    "Great tool for productivity, highly recommend it.",
    "Not worth the price, very disappointing experience.",
]


def load_model(model_path: str):
    """Load the pre-trained BERT model from a saved state dict."""
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(text: str, model, tokenizer):
    """Tokenize input text and run inference. Returns label and confidence."""
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    label = "Positive" if predicted_class == 1 else "Negative"
    return label, confidence


def main():
    print("Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.\n")

    print("=" * 60)
    print("BERT Sentiment Analysis Demo — Amazon Software Reviews")
    print("=" * 60)

    for i, review in enumerate(SAMPLE_REVIEWS, start=1):
        label, confidence = predict(review, model, tokenizer)
        print(f"\nReview {i}: {review}")
        print(f"  Predicted Sentiment : {label}")
        print(f"  Confidence          : {confidence:.4f} ({confidence * 100:.2f}%)")

    print("\n" + "=" * 60)
    print("Demo complete.")


if __name__ == "__main__":
    main()

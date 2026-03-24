# Final Report: LSTM Model for Sentiment Analysis (Draft by Xuyang)

## 1. Data Preprocessing & Embeddings
For the LSTM baseline, the raw Amazon reviews were converted explicitly to lower-case and stripped of punctuation. To reduce noise, we utilized NLTK's English stopword corpus to filter out non-informative words during tokenization, significantly improving feature saliency. 
To build the numerical representation, we utilized pre-trained GloVe embeddings (300-dimensional from the 6B corpus). A mapping was created linking our dataset's training vocabulary (16,199 words) directly to the corresponding GloVe vectors. Out-of-vocabulary (OOV) tokens were handled via a normal distribution initialization.

## 2. Model Architecture
The neural architecture consists of an Embedding layer followed by a 2-layer Long Short-Term Memory (LSTM) network.
- **Embedding Layer:** Fixed to `300` dimensions and initialized with the GloVe matrix weights. This layer was frozen (`requires_grad=False`) to prevent overfitting on our subset of 10K dataset instances.
- **LSTM Layers:** Two stacked LSTM layers with a hidden dimension of `256`, leveraging a `0.5` Dropout probability between layers to enhance generalization and limit codependence.
- **Classification Head:** The last hidden state passes through a fully connected (`Linear`) layer with a `Sigmoid` activation function, outputting a continuous probability suited for binary classification.

## 3. Training Process
The model was trained using Mac MPS GPU hardware acceleration with the `Adam` optimizer and Binary Cross Entropy Loss (`BCELoss`). We used a batch size of 64 and a learning rate of 1e-3.
- To prevent overfitting, early stopping monitored validation loss with a patience of 3. The validation loss minimized at Epoch 4 ($\approx$0.497), and training was halted at Epoch 7 as validation performance continued to deteriorate. We restored the optimal model weights prior to testing.

## 4. Evaluation and Results
Evaluated on the 10% held-out test set (1,121 reviews), the LSTM produced the following performance metrics:
- **Test Accuracy:** 80.37%
- **F1-Score:** 0.8907
- **Precision:** 80.50%
- **Recall:** 99.67%
- **Confusion Matrix:** 896 True Positives, 5 True Negatives, 217 False Positives, 3 False Negatives.

### Class Imbalance Analysis
The confusion matrix reveals a severe class imbalance in the dataset. Of the 1,121 test samples, 899 (80.2%) are positive and only 222 (19.8%) are negative. This imbalance propagates from the raw Amazon review distribution, where high-star ratings are far more common than low-star ratings in the Software category.

As a result, the model learned a heavily positive-biased decision boundary: it correctly identifies 896 out of 899 positive reviews (99.67% recall on positive class) but only identifies 5 out of 222 negative reviews (2.25% recall on negative class). In other words, the model classifies nearly all inputs as positive regardless of actual sentiment.

### Discussion
The 80.37% accuracy appears acceptable at first glance, but is largely explained by the class prior — a trivial classifier that always predicts positive would achieve ~80.2% accuracy. The more informative metric is the near-zero true negative rate, which indicates the model has not meaningfully learned to distinguish negative sentiment.

This performance gap relative to the expected ~85% accuracy can be attributed to two factors: (1) the class imbalance was not addressed during training (standard BCELoss weights all samples equally, so the model minimizes loss by favoring the majority class), and (2) static GloVe embeddings may fail to capture the nuanced and often sarcastic language patterns present in negative software reviews.

A natural mitigation would be to use a weighted loss function (e.g., `pos_weight` in `BCEWithLogitsLoss`) to penalize false negatives more heavily. This remains a direction for future improvement and provides a clear motivation for evaluating BERT, whose contextual embeddings may better handle the linguistic complexity of negative reviews.

# Sentiment Analysis of Product Reviews

This project performs sentiment analysis on product reviews using two deep learning models: **LSTM** (Keras) and **BERT** (Hugging Face Transformers). The goal is to classify reviews as **Positive** or **Negative** based on the review text.

## 📁 Project Structure

```
sentiment-analysis/
│
├── models/                       # Saved models
│   ├── best_lstm_model.keras
│   └── best_bert_model.pt
│
├── images/                       # Visualizations
│   ├── wordcloud_positive.png
│   ├── wordcloud_negative.png
│   ├── lstm_confusion_matrix.png
│   └── bert_confusion_matrix.png
│
├── notebook.ipynb                # Jupyter notebook with all code
├── requirements.txt              # List of required packages
└── README.md                     # Project documentation
```

## 📊 Dataset

- **Source**: Amazon Product Reviews dataset (Video Games category)
- **Size**: 100,000+ reviews (subset used for LSTM and BERT)
- **Labels**: Positive (rating > 3), Negative (rating ≤ 3)

## 🧠 Models Used

### 1. LSTM (Keras)
- Tokenization using Keras `Tokenizer`
- Padding & truncation to fixed length (e.g., 100)
- LSTM + Dense + Softmax layers
- Training with `binary_crossentropy`

### 2. BERT (Transformers)
- Pretrained model: `bert-base-uncased`
- Tokenization using `BertTokenizer`
- Fine-tuned using Hugging Face `Trainer` API or custom loop
- Optimizer: AdamW, Scheduler: LinearWarmup

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| LSTM  | ~0.88    | ~0.88     | ~0.87  | ~0.87    |
| BERT  | ~0.91    | ~0.91     | ~0.90  | ~0.90    |

- BERT outperformed LSTM with slightly higher accuracy and F1-score.
- Both models show strong performance on this binary classification task.

## 📉 Visualizations

- WordClouds for positive and negative reviews.
- Confusion matrices for both models.
- Classification reports printed in the notebook.

## 💾 Saved Models

- `models/best_lstm_model.keras`: Trained LSTM model (Keras format)
- `models/best_bert_model.pt`: Trained BERT model (PyTorch format)

## ✅ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

1. Clone the repo
2. Open `notebook.ipynb` in Jupyter
3. Run all cells in order to train, evaluate, and save models

## ✍️ Author

Eugene Kwesi Acquah — Data Science Student & NLP Enthusiast

## 📚 References

- Hugging Face Transformers
- Keras & TensorFlow Documentation
- scikit-learn Metrics
- Kaggle Amazon Reviews Dataset
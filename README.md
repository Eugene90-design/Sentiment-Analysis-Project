# Sentiment Analysis of Amazon Product Reviews

This project applies Natural Language Processing (NLP) techniques to classify product reviews as either positive or negative. Two models were implemented: a deep learning-based LSTM model and a pre-trained BERT transformer model. Evaluation was performed using classification metrics and visualizations such as confusion matrices and word clouds.

## 📁 Dataset
- **Source:** Amazon Product Reviews (Video Games category)
- **Format:** `.tsv` file with two columns: `text` (review), `label` (1 = positive, 0 = negative)

## 🧠 Models Used

### 1. LSTM Model (Keras)
- Tokenization and padding with TensorFlow/Keras.
- Embedding layer, LSTM layer, and Dense output.
- Trained for binary classification.

### 2. BERT Model (Transformers)
- Pre-trained `bert-base-uncased` model from Hugging Face.
- Fine-tuned using PyTorch and Hugging Face `Trainer`.
- Used `BertTokenizer` for tokenization.

## 📊 Evaluation
- **Metrics:** Accuracy, Precision, Recall, F1-score.
- **Visualizations:** Word clouds and confusion matrices.

### Sample Results:
- **LSTM Accuracy:** ~0.85
- **BERT Accuracy:** ~0.91

![LSTM Confusion Matrix](IMAGES/lstm_confusion_matrix.png)
![BERT Confusion Matrix](IMAGES/bert_confusion_matrix.png)

## 📦 Folder Structure
```
.
├── SentimentAnalysis.ipynb
├── README.md
├── requirements.txt
├── IMAGES/
│   ├── wordcloud_positive.png
│   ├── wordcloud_negative.png
│   ├── lstm_confusion_matrix.png
│   └── bert_confusion_matrix.png
├── models/
│   ├── best_lstm_model.keras
│   └── best_bert_model.pt
```

## ✅ How to Run

1. Clone this repo:
```bash
git clone https://github.com/yourusername/sentiment-analysis-bert-lstm.git
cd sentiment-analysis-bert-lstm
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Open the notebook and run all cells:
```bash
jupyter notebook SentimentAnalysis.ipynb
```

## 🧾 Requirements
See `requirements.txt` for full list. Main libraries:
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- tensorflow
- torch
- transformers
- wordcloud

## 📚 License
MIT License
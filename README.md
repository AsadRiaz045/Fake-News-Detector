# 📰 Fake News Detector

A Machine Learning based web application that detects whether a news article is **REAL** or **FAKE**. This project utilizes a fine-tuned **BERT model** to analyze text and predict its authenticity with high accuracy.

## 🚀 Features
- **Real-time Detection:** Enter any news headline or article to get instant results.
- **High Accuracy:** Powered by the BERT (Bidirectional Encoder Representations from Transformers) model.
- **User-Friendly Interface:** Simple and easy-to-use web interface.

## 🛠️ Tech Stack
- **Language:** Python 🐍
- **Model:** BERT (Hugging Face Transformers)
- **Framework:** Flask / Streamlit (Select one based on your app.py)
- **Libraries:** PyTorch, Scikit-learn, Pandas, NumPy

## 📂 Project Structure
```text
├── my_bert_model/          # Pre-trained model files (Large files ignored in git)
├── app.py                  # Main application file
├── config.json             # Model configuration
├── vocab.txt               # BERT vocabulary
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation

# 📰 Fake News Detector

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fake-news-detector-rfys6vblfjletwunvmvdcz.streamlit.app/)

A Machine Learning based web application that detects whether a news article is **REAL** or **FAKE**. This project utilizes a fine-tuned **DistilBERT model** to analyze text and predict its authenticity with high accuracy.

## 🚀 Live Demo
Check out the live application here: **[Click to View App](https://fake-news-detector-rfys6vblfjletwunvmvdcz.streamlit.app/)**

## ✨ Features
- **Real-time Detection:** Enter any news headline or article to get instant results.
- **High Accuracy:** Powered by the **DistilBERT** (Distilled Bidirectional Encoder Representations from Transformers) model.
- **Confidence Score:** Displays the probability percentage of the news being Real or Fake.
- **User-Friendly Interface:** Built with Streamlit for a clean and responsive experience.

## 🛠️ Tech Stack
- **Language:** Python 🐍
- **Model:** DistilBERT (Hugging Face Transformers)
- **Frontend:** Streamlit
- **Libraries:** PyTorch, Scikit-learn, NumPy
- **Hosting:** Streamlit Cloud (App) & Hugging Face (Model)

## 📂 Project Structure
```text
├── app.py                  # Main Streamlit application
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
└── .gitignore             # Files to ignore
## ⚙️ How to Run Locally

If you want to run this project on your local machine, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/AsadRiaz045/Fake-News-Detector.git
   cd Fake-News-Detector
2.**Install dependencies**
pip install -r requirements.txt
3.**run the app**
streamlit run app.py

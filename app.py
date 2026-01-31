import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import time

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️", layout="centered")

# CSS Design
st.markdown("""
    <style>
    .stTextArea textarea {font-size: 16px;}
    .stButton button {background-color: #FF4B4B; color: white; font-size: 18px; width: 100%;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODEL (CORRECT PATH) ---
@st.cache_resource
def load_model():
    # Model wahi dhoondo jahan app.py parri hai
    model_path ="Asadriaz525/fake-news-detector"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

tokenizer, model = load_model()

# --- 3. APP INTERFACE ---
st.title("🕵️ Fake News Detector ")
st.markdown("### Paste any news article below to verify its authenticity.")
st.write("---")

news_text = st.text_area("📰 News Text:", height=200, placeholder="Type or paste news article here...")

if st.button("🔍 Check Authenticity"):
    if not news_text:
        st.warning("⚠️ Please enter some text first!")
    elif model is None:
        st.error("Model files not found! Please check folder structure.")
    else:
        # Animation (Thora Real lagne ke liye)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🧠 Reading text...")
        progress_bar.progress(30)
        time.sleep(0.3)
        
        status_text.text("🤖 Analyzing patterns with BERT...")
        progress_bar.progress(70)
        
        # --- PREDICTION LOGIC ---
        inputs = tokenizer(news_text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            
        # Result Extraction
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item() * 100
        
        # Probabilities for Chart
        prob_0 = probabilities[0][0].item() * 100
        prob_1 = probabilities[0][1].item() * 100
        
        progress_bar.progress(100)
        status_text.empty()

        # --- DISPLAY RESULT (SWAPPED LOGIC) ---
        st.divider()
        
        # Yahan humne logic badal di hai taake sahi jawab aye
        # Agar Prediction 0 hai -> TO REAL NEWS HAI
        if prediction == 0:
            st.success(f"## ✅ REAL NEWS")
            st.caption(f"Confidence: {confidence:.2f}%")
            st.balloons()
        
        # Agar Prediction 1 hai -> TO FAKE NEWS HAI
        else:
            st.error(f"## 🚨 FAKE NEWS")
            st.caption(f"Confidence: {confidence:.2f}%")
        
        # Detailed Analytics
        st.write("### 📊 AI Analysis")
        col1, col2 = st.columns(2)
        
        # Note: Chart labels bhi update kar diye hain
        with col1:
            st.metric(label="Real Probability", value=f"{prob_0:.1f}%")
        with col2:
            st.metric(label="Fake Probability", value=f"{prob_1:.1f}%")
            
        # Bar Chart showing Real Probability
        st.progress(int(prob_0))
        st.caption("Blue bar indicates chance of being 'Real'.")

# --- FOOTER ---
st.markdown("---")

st.markdown("🚀 *Powered by DistilBERT & Streamlit* | Developed by Asad")




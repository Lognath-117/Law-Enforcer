import streamlit as st
import pandas as pd
import pickle
import re
import nltk
import base64

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from googletrans import Translator

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Indian Law Submitter",
    page_icon="⚖️",
    layout="centered"
)

# -------------------- NLTK SETUP --------------------
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------- BACKGROUND IMAGE (PNG) --------------------
def set_background(png_file):
    with open(png_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("bg.png")

# -------------------- LOAD TRAINED MODEL FILES --------------------
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("law_vectors.pkl", "rb") as f:
    law_vectors = pickle.load(f)

df = pd.read_pickle("ipc_dataframe.pkl")

# -------------------- TRANSLATOR --------------------
translator = Translator()

def translate_to_english(text):
    try:
        lang = detect(text)
        if lang != "en":
            text = translator.translate(text, dest="en").text
    except:
        pass
    return text

# -------------------- TEXT PREPROCESSING --------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)

    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)

# -------------------- PREDICTION FUNCTION --------------------
def predict_laws(crime_text):
    # Multilingual handling
    crime_text = translate_to_english(crime_text)

    processed = preprocess(crime_text)
    user_vector = vectorizer.transform([processed])

    similarity = cosine_similarity(user_vector, law_vectors)[0]
    top_indices = similarity.argsort()[-3:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "section": df.iloc[idx]['Section'],
            "description": df.iloc[idx]['Description']
        })
    return results

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #f5d27a;
    text-shadow: 2px 2px 4px #000;
    margin-top: 30px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: white;
    margin-bottom: 25px;
}

.law-card {
    background: rgba(255,255,255,0.25);
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 15px;
    color: black;
}

.law-card h4,
.law-card p {
    color: black;
}

/* Running footer */
.footer-container {
    width: 100%;
    overflow: hidden;
    margin-top: 35px;
}

.footer-text {
    display: inline-block;
    white-space: nowrap;
    font-size: 18px;
    color: #f5d27a;
    animation: scroll-left 12s linear infinite;
}

@keyframes scroll-left {
    0% { transform: translateX(100%); }
    100% { transform: translateX(-100%); }
}
</style>
""", unsafe_allow_html=True)

# -------------------- UI --------------------
st.markdown('<div class="title">Indian Law Submitter</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multilingual NLP-Based Legal Section Recommendation</div>', unsafe_allow_html=True)

crime_text = st.text_area(
    "Enter Crime Description (Any Language):",
    placeholder="English / தமிழ் / हिंदी",
    height=150
)

if st.button("Submit"):
    if crime_text.strip() == "":
        st.warning("Please enter a crime description.")
    else:
        results = predict_laws(crime_text)

        st.markdown("<h3 style='color:white'>Applicable Indian Laws</h3>", unsafe_allow_html=True)

        for law in results:
            st.markdown(f"""
            <div class="law-card">
                <h4>{law['section']}</h4>
                <p>{law['description']}</p>
            </div>
            """, unsafe_allow_html=True)

# -------------------- RUNNING FOOTER --------------------
st.markdown("""
<div class="footer-container">
    <div class="footer-text">⚖️ Project by Lognath ⚖️</div>
</div>
""", unsafe_allow_html=True)

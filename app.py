import streamlit as st
import pandas as pd
import pickle
import re
import nltk
import base64
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Indian Law Submitter",
    page_icon="⚖️",
    layout="centered"
)

# -------------------- NLTK SETUP --------------------
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------- LOAD BACKGROUND IMAGE (PNG) --------------------
def set_bg(png_file):
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

set_bg("bg.png")

# -------------------- LOAD TRAINED MODEL --------------------
with open(os.path.join("models", "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

with open(os.path.join("models", "law_vectors.pkl"), "rb") as f:
    law_vectors = pickle.load(f)

df = pd.read_pickle(os.path.join("models", "ipc_dataframe.pkl"))

# -------------------- PREPROCESSING --------------------
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z ]', '', text)

    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# -------------------- PREDICTION --------------------
def predict_laws(crime_text):
    processed = preprocess(crime_text)
    user_vector = vectorizer.transform([processed])
    similarity = cosine_similarity(user_vector, law_vectors)[0]
    top_indices = similarity.argsort()[-3:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "section": df.iloc[idx]['Section'],
            "description": df.iloc[idx]['Description'],
            "score": round(similarity[idx], 2)
        })
    return results

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
.main-card {
    max-width: 900px;
    margin: auto;
    background: rgba(0,0,0,0.65);
    padding: 35px;
    border-radius: 18px;
    box-shadow: 0 0 30px rgba(0,0,0,0.6);
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #f5d27a;
    text-shadow: 2px 2px 4px #000;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: white;
    margin-bottom: 25px;
}

.law-card {
    background: rgba(255,255,255,0.92);
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 15px;
}

.footer {
    text-align: center;
    color: #ddd;
    margin-top: 30px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- UI --------------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown('<div class="title">Indian Law Submitter</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">NLP-Based Legal Section Recommendation</div>', unsafe_allow_html=True)

crime_text = st.text_area(
    "Enter Crime Description:",
    placeholder='e.g., "Someone fraudulently took money promising a job and never returned it."',
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
                <b>Confidence Score:</b> {law['score']}
            </div>
            """, unsafe_allow_html=True)

st.markdown('<div class="footer">Project by Lognath</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


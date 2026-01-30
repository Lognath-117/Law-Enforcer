import gradio as gr
import pandas as pd
import pickle
import re
import nltk
import base64
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- NLTK SETUP --------------------
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------- LOAD BACKGROUND IMAGE (PNG) --------------------
BG_PATH = os.path.join("assets", "bg.png")
with open(BG_PATH, "rb") as img:
    encoded_bg = base64.b64encode(img.read()).decode()

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
    if not crime_text.strip():
        return "<p style='color:white'>Please enter a valid crime description.</p>"

    processed = preprocess(crime_text)
    user_vector = vectorizer.transform([processed])
    similarity = cosine_similarity(user_vector, law_vectors)[0]
    top_indices = similarity.argsort()[-3:][::-1]

    cards = ""
    for idx in top_indices:
        cards += f"""
        <div class="law-card">
            <h3>{df.iloc[idx]['Section']}</h3>
            <p>{df.iloc[idx]['Description']}</p>
            <span>Confidence Score: {round(similarity[idx], 2)}</span>
        </div>
        """

    return cards

# -------------------- UI CSS (PNG BACKGROUND) --------------------
custom_css = f"""
body {{
    background-image: url("data:image/png;base64,{encoded_bg}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

#container {{
    max-width: 900px;
    margin: auto;
    margin-top: 40px;
}}

#title {{
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #f5d27a;
    text-shadow: 2px 2px 4px #000;
}}

#subtitle {{
    text-align: center;
    font-size: 18px;
    color: white;
    margin-bottom: 30px;
}}

#input_box textarea {{
    font-size: 16px;
    padding: 14px;
    border-radius: 10px;
}}

#submit_btn {{
    display: block;
    margin: 20px auto;
    background: linear-gradient(to right, #1e3c72, #2a5298);
    color: white;
    font-size: 16px;
    border-radius: 8px;
}}

.law-card {{
    background: rgba(255,255,255,0.92);
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 15px;
}}

.law-card h3 {{
    margin: 0;
    color: #1e3c72;
}}

.law-card span {{
    color: #333;
    font-weight: bold;
}}

#footer {{
    text-align: center;
    color: #ddd;
    margin-top: 30px;
    font-size: 14px;
}}
"""

# -------------------- GRADIO UI --------------------
with gr.Blocks(css=custom_css) as demo:

    with gr.Column(elem_id="container"):
        gr.Markdown("<div id='title'>Indian Law Submitter</div>")
        gr.Markdown("<div id='subtitle'>NLP-Based Legal Section Recommendation</div>")

        crime_input = gr.Textbox(
            lines=6,
            placeholder='e.g., "Someone fraudulently took money promising a job and never returned it."',
            label="Enter Crime Description",
            elem_id="input_box"
        )

        submit_btn = gr.Button("Submit", elem_id="submit_btn")

        law_output = gr.Markdown()

        submit_btn.click(
            fn=predict_laws,
            inputs=crime_input,
            outputs=law_output
        )

        gr.Markdown("<div id='footer'>Project by Lognath</div>")

demo.launch(server_name="127.0.0.1", server_port=7860)

import os
from dotenv import load_dotenv
from flask import Flask, request, render_template
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from vertexai.language_models import TextEmbeddingModel

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

load_dotenv()

app = Flask(__name__)

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = "us-central1" 
EMBEDDINGS_OUTPUT_FILE = "document_embeddings.json"

if not PROJECT_ID:
    print("Error: GCP_PROJECT_ID environment variable not set.")
    exit(1)
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    print("Error: GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
    exit(1)

# Alustetaan Vertex AI SDK. Tämä on tärkeä vaihe!
vertexai.init(project=PROJECT_ID, location=REGION)

# Alustetaan mallit kerran sovelluksen käynnistyessä tehokkuuden parantamiseksi
EMBEDDING_MODEL = TextEmbeddingModel.from_pretrained("text-embedding-004")
GENERATIVE_MODEL = GenerativeModel(model_name="gemini-2.0-flash")

THEMED_DOC_DATA = {}
ALL_THEMES = []

def load_document_data(filepath=EMBEDDINGS_OUTPUT_FILE):
    """Lataa upotukset JSON-tiedostosta ja järjestää ne teemoittain."""
    global THEMED_DOC_DATA, ALL_THEMES
    if not os.path.exists(filepath):
        print(f"Error: Document embeddings file '{filepath}' not found. Please run create_embeddings.py first.")
        exit(1)
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            theme = item.get("theme", "yleinen")
            if theme not in THEMED_DOC_DATA:
                THEMED_DOC_DATA[theme] = {"embeddings": [], "texts": []}
                ALL_THEMES.append(theme)
            THEMED_DOC_DATA[theme]["embeddings"].append(item["embedding"])
            THEMED_DOC_DATA[theme]["texts"].append(item["text"])
    
    print(f"Loaded embeddings for themes: {ALL_THEMES}")
    ALL_THEMES.sort()

load_document_data()

def get_embedding(text):
    """Kutsuu Vertex AI Embeddings APIa tekstin upotuksen luomiseksi."""
    embeddings = EMBEDDING_MODEL.get_embeddings([text])
    return embeddings[0].values

def get_llm_response(prompt):
    """Kutsuu Vertex AI Gemini-pro-mallia vastauksen generointiin."""
    try:
        response = GENERATIVE_MODEL.generate_content(
            [Part.from_text(prompt)],
            generation_config={"temperature": 0.2}
        )
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Valitettavasti tekoälyltä ei saatu vastausta. Yritä myöhemmin uudelleen."


@app.route('/')
def index():
    """Näyttää chatbotin käyttöliittymän."""
    return render_template('index.html', themes=ALL_THEMES)

@app.route('/ask', methods=['POST'])
def ask_chatbot():
    """Käsittelee käyttäjän kyselyn, suorittaa RAG-haun ja palauttaa vastauksen."""
    user_query = request.form['query']
    selected_theme = request.form['theme']

    if selected_theme in THEMED_DOC_DATA:
        current_doc_embeddings = THEMED_DOC_DATA[selected_theme]["embeddings"]
        current_doc_texts = THEMED_DOC_DATA[selected_theme]["texts"]
    else:
        current_doc_embeddings = []
        current_doc_texts = []
        print(f"Warning: Selected theme '{selected_theme}' not found in loaded data.")
    
    if not current_doc_embeddings:
        chatbot_response = "Valitettavasti valitusta teemasta ei löytynyt tietoa."
        return render_template('index.html', query=user_query, response=chatbot_response, themes=ALL_THEMES, selected_theme=selected_theme)

    query_embedding = get_embedding(user_query)

    doc_embeddings_np = np.array(current_doc_embeddings)
    query_embedding_np = np.array(query_embedding).reshape(1, -1)

    similarities = cosine_similarity(query_embedding_np, doc_embeddings_np)[0]
    top_n_indices = np.argsort(similarities)[::-1][:3]

    context = ""
    for i in top_n_indices:
        context += current_doc_texts[i] + "\n\n"

    llm_prompt = f"""
    Käyttäjän kysymys: {user_query}

    Käytä tätä tietoa vastataksesi kysymykseen, yritä välttää sanaa kojnteksti vastauksessa:
    {context}

    Vastaa käyttäjän kysymykseen yllä olevan kontekstin perusteella. Jos konteksti ei sisällä tarvittavaa tietoa, ilmoita, ettet löytänyt vastausta.
    Anna vastaus selkeästi, yritä välttää vastauksessa mainintaa viittauksesta kontekstiin, jossei sitä erityisesti pyydetä (älä käytä esimerkiksi "kontekstin perusteella" kaltaisia ilmauksia),
    käytä esimerkiksi "Valitettavasti valitusta teemasta ei löytynyt tietoa." tyylistä lausetta.
    """
    chatbot_response = get_llm_response(llm_prompt)

    return render_template('index.html', query=user_query, response=chatbot_response, themes=ALL_THEMES, selected_theme=selected_theme)

if __name__ == '__main__':
    app.run(debug=True)

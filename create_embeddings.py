import os
from dotenv import load_dotenv
from google.cloud import storage
import vertexai
from vertexai.language_models import TextEmbeddingModel
from docx import Document
import json
import tempfile

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = "us-central1"
GCS_BUCKET_NAME = "chatbot-57488-documents" # Korvaa omalla bucketin nimellä
EMBEDDINGS_OUTPUT_FILE = "document_embeddings.json"

if not PROJECT_ID:
    print("Error: GCP_PROJECT_ID environment variable not set.")
    exit(1)
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    print("Error: GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
    exit(1)

vertexai.init(project=PROJECT_ID, location=REGION)
storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(GCS_BUCKET_NAME)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

def get_embedding(text):
    """Kutsuu Vertex AI Embeddings APIa tekstin upotuksen luomiseksi."""
    # TypeError: ... got an unexpected keyword argument 'task_type'
    # Korjataan poistamalla vanhentuneet parametrit. Uudempi SDK-versio ei enää
    # käytä 'task_type' tai 'auto_truncate' -parametreja tällä tavalla.
    embeddings = embedding_model.get_embeddings([text])
    return embeddings[0].values

def extract_text_from_docx_blob(blob):
    """Lataa docx-tiedosto GCS:stä ja purkaa tekstin."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
            blob.download_to_filename(temp_file.name)
            temp_filepath = temp_file.name
        
        document = Document(temp_filepath)
        full_text = []
        for para in document.paragraphs:
            full_text.append(para.text)
        
        os.remove(temp_filepath)
        return '\n'.join(full_text)
    except Exception as e:
        print(f"Error extracting text from DOCX {blob.name}: {e}")
        return None

def process_documents():
    """Käsittelee GCS-bucketin dokumentit, luo upotukset ja tallentaa ne JSON-tiedostoon."""
    all_document_data = []
    blobs = bucket.list_blobs()

    for blob in blobs:
        if blob.name.endswith('/'):
            continue

        content = None
        theme = blob.name.split('/')[0] if '/' in blob.name else "yleinen"

        if blob.name.endswith(".txt"):
            print(f"Processing TXT file: {blob.name} (Theme: {theme})...")
            content = blob.download_as_text()
        elif blob.name.endswith(".docx"):
            print(f"Processing DOCX file: {blob.name} (Theme: {theme})...")
            content = extract_text_from_docx_blob(blob)
        else:
            print(f"Skipping unsupported file type: {blob.name}")
            continue

        if content:
            chunk_size = 500
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    embedding = get_embedding(chunk)
                    if embedding:
                        all_document_data.append({
                            "source": blob.name,
                            "theme": theme,
                            "chunk_index": i,
                            "text": chunk,
                            "embedding": embedding
                        })
    
    with open(EMBEDDINGS_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_document_data, f, ensure_ascii=False, indent=2)
    print(f"Embeddings saved to {EMBEDDINGS_OUTPUT_FILE}")

if __name__ == "__main__":
    process_documents()

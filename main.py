import os
from dotenv import load_dotenv

# Lataa ympäristömuuttujat .env-tiedostosta
load_dotenv()

# Nämä muuttujat ovat nyt käytettävissä os.getenv() -kutsuilla
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = "europe-west1" # Valitse sopiva alue, esim. "us-central1" tai "europe-west1"

# Alusta Vertex AI client. Tämä käyttää automaattisesti GOOGLE_APPLICATION_CREDENTIALS-muuttujaa.
from google.cloud import aiplatform
aiplatform.init(project=PROJECT_ID, location=REGION)


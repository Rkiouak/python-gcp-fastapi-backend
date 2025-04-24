python3 -m venv env
source env/bin/activate

fastapi dev main.py

gcloud run deploy python-gcp-fastapi-backend --source . --region us-central1 --allow-unauthenticated

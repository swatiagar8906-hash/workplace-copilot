import os

import requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GROQ_API_KEY")
if not key:
    raise SystemExit("GROQ_API_KEY nahi mili. .env file check karo.")

r = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {key}"},
    timeout=30,
)
r.raise_for_status()
for m in sorted(r.json()["data"], key=lambda m: m["id"]):
    print(m["id"])

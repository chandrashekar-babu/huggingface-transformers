from huggingface_hub import login, list_models
from dotenv import load_dotenv
import os


load_dotenv("../dotenv.sh")
HF_TOKEN = os.getenv("HF_TOKEN")

# Log in to Hugging Face Hub
login(token=HF_TOKEN)

# WARNING: This will fetch and load all models (2.5 million+) from the Hugging Face Hub, which may take a long time and consume a lot of memory.
#for model in list_models():
#    print(f"Model: {model.modelId}, Downloads: {model.downloads:,}")


for model in list_models(filter="text-classification", sort="downloads", limit=10):
    print(f"Model: {model.modelId}, Downloads: {model.downloads:,}" )
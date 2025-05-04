from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import pipeline

# Load Hugging Face sentiment pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

app = FastAPI()

# Define the expected request body
class TextRequest(BaseModel):
    text: str

device = torch.device("cpu")

# Load the GPT-2 text generation model
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr", device=-1)

@app.post("/sentiment_analysis")
async def sentiment_analysis(request: TextRequest):
    result = sentiment_pipeline(request.text)[0]
    return {
        "text": request.text,
        "label": result['label'],
        "score": result['score']
    }


class QARequest(BaseModel):
    question: str

@app.post("/translation_en_to_fr")
async def generate_answer(request: QARequest):
    result = translator(request.question, max_length=50, truncation=True)[0]
    return {"answer": result['translation_text']}

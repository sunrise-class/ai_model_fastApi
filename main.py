from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Load Hugging Face sentiment pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

app = FastAPI()

# Define the expected request body
class TextRequest(BaseModel):
    text: str

@app.post("/openai")
async def sentiment_analysis(request: TextRequest):
    result = sentiment_pipeline(request.text)[0]
    return {
        "text": request.text,
        "label": result['label'],
        "score": result['score']
    }

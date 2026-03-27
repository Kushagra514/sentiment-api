from fastapi import FastAPI
from pydantic import BaseModel
from src.sentiment_api.model import train_model, predict_sentiment

app = FastAPI()

@app.get("/health")
def health():
    return {"status" : "ok"}

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    model,vectorizer = train_model()
    sentiment = predict_sentiment(input.text,model,vectorizer)
    return {"sentiment": sentiment}
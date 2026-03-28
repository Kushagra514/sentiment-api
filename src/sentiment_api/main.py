from fastapi import FastAPI
from pydantic import BaseModel
from src.sentiment_api.model import train_model, predict_sentiment, load_model, save_model
from contextlib import asynccontextmanager

ml_models = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    model, vectorizer = train_model()
    save_model(model, vectorizer)
    ml_models["model"], ml_models["vectorizer"] = load_model()
    yield
    ml_models.clear()

app = FastAPI(lifespan = lifespan)

@app.get("/health")
def health():
    return {"status" : "ok"}

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    model,vectorizer = ml_models["model"],ml_models["vectorizer"]
    sentiment = predict_sentiment(input.text,model,vectorizer)
    return {"sentiment": sentiment}
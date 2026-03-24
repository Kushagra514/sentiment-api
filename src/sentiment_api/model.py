from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

#training data
TEXTS = [
    "I love this product",
    "This is amazing",
    "Great experience",
    "I hate this",
    "Terrible product",
    "Very disappointing"
]

LABELS = ["positive","positive","positive","negative","negative","negative"]

def train_model():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(TEXTS)
    model = LogisticRegression()
    model.fit(X,LABELS)
    return model,vectorizer

def predict_sentiment(text: str,model,vectorizer) -> str:
    X = vectorizer.transform([text])
    return model.predict(X)[0]

def save_model(model, vectorizer, path: str = "model.pkl"):
    with open(path, "wb") as f:
        pickle.dump((model,vectorizer),f)

def load_model(path: str = "model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
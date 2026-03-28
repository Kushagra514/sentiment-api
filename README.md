# sentiment-api

A production-grade sentiment analysis REST API built with FastAPI, scikit-learn, and Docker. Deployed live on Render.

## Live Demo
**API URL:** https://sentiment-api-3oke.onrender.com  
**Swagger Docs:** https://sentiment-api-3oke.onrender.com/docs

## Stack
- **FastAPI** — REST API framework
- **scikit-learn** — TF-IDF vectorizer + Logistic Regression model
- **Docker** — containerization
- **Render** — cloud deployment
- **Poetry** — dependency management

## Endpoints
- `GET /health` — health check
- `POST /predict` — takes a sentence, returns positive or negative sentiment

## Run Locally
```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

## Example
```json
POST /predict
{"text": "I love this product"}
→ {"sentiment": "positive"}
```
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ml.model import model
from kafka_client import log_user_event

app = FastAPI()

class RecommendationRequest(BaseModel):
    movie_title: str
    user_id: int

@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    try:
        recommendations = model.get_recommendations(request.movie_title)
        log_user_event(request.user_id, request.movie_title, "search")
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

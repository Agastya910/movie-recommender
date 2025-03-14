from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from kafka_client import produce_message, consume_messages
import asyncio

app = FastAPI()

# Pydantic model for request body
class RecommendationRequest(BaseModel):
    query: str

# Endpoint to generate recommendations
@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    query = request.query

    # Send query to Kafka topic
    produce_message("recommendation_requests", {"query": query})

    # Simulate processing delay
    await asyncio.sleep(2)

    # Consume recommendation from Kafka topic
    for message in consume_messages("recommendation_results"):
        if message.get("query") == query:
            return {"recommendation": message.get("recommendation")}

    raise HTTPException(status_code=404, detail="Recommendation not found")

# Background task to process recommendations
async def process_recommendations():
    for message in consume_messages("recommendation_requests"):
        query = message.get("query")
        # Simulate recommendation generation (replace with your logic)
        recommendation = f"Recommended movies for '{query}'"
        produce_message("recommendation_results", {"query": query, "recommendation": recommendation})

# Start background task when the app starts
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_recommendations())

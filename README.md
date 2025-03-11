# End-to-End Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.9-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.85.0-green)
![Kafka](https://img.shields.io/badge/Kafka-3.2.0-orange)
![Docker](https://img.shields.io/badge/Docker-20.10.12-blue)
![AWS](https://img.shields.io/badge/AWS-EC2%2C%20ECR-yellow)

## Overview

This project is an **end-to-end movie recommendation system** built using **Python**, **FastAPI**, **Kafka**, **Docker**, and **AWS (EC2, ECR)**. The system leverages **BM25** and **Sentence Transformers** for embedding data, and it supports **online learning** through a Kafka pipeline for real-time data streaming. The FastAPI backend handles user queries and serves recommendations.

### Key Features
- **Real-time Recommendations**: Uses Kafka for streaming user interactions and updating recommendations dynamically.
- **Online Learning**: Continuously improves the recommendation model based on user interactions.
- **Scalable Architecture**: Deployed on AWS EC2 with Docker for easy scaling and management.
- **Sentence Transformers**: Generates semantic embeddings for movie titles and descriptions.
- **FastAPI Backend**: Provides a RESTful API for querying recommendations.

---

## Architecture

The system is divided into three main components:

1. **Data Ingestion & Streaming**:
   - **Kafka**: Handles real-time user events (e.g., clicks, ratings).
   - **Zookeeper**: Manages Kafka brokers.

2. **Model Training & Online Learning**:
   - **Sentence Transformers**: Generates embeddings for movie titles.
   - **BM25**: Used for ranking and retrieval.
   - **Online Learning**: Updates user profiles and recommendations based on Kafka events.

3. **Model Serving & API**:
   - **FastAPI**: Exposes the recommendation model as a REST API.
   - **Docker**: Containerizes the application for deployment.
   - **AWS EC2**: Hosts the application in the cloud.

---

## Setup Instructions

### Prerequisites
- Python 3.9
- Docker
- Docker Compose
- AWS Account (for EC2 and ECR)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Agastya910/movie-recommender.git
cd movie-recommender
```

### Step 2: Install Dependencies
```bash

pip install -r requirements.txt
```

### Step 3: Set Up Kafka and Zookeeper
Start Kafka and Zookeeper using Docker Compose:

```bash

docker-compose up --build
Verify Kafka is running:
```
```bash

docker exec -it movie-recommender_kafka_1 kafka-topics --list --bootstrap-server localhost:9092
```

### Step 4: Generate Embeddings
Run the following script to generate embeddings for the movie dataset:

```bash

python ml/generate_embeddings.py
```

### Step 5: Start the FastAPI Service
Build and run the FastAPI service:

```bash

docker-compose up --build
Access the API documentation at:
Copy
http://localhost:8000/docs
```

### Step 6: Deploy to AWS EC2
Push Docker Image to ECR:

```bash

aws ecr create-repository --repository-name movie-recommender
docker tag movie-recommender_app:latest <aws_account_id>.dkr.ecr.<region>.amazonaws.com/movie-recommender:latest
docker push <aws_account_id>.dkr.ecr.<region>.amazonaws.com/movie-recommender:latest
```
Deploy to EC2:

Launch an EC2 instance with Docker installed.

Pull the Docker image:

```bash

docker pull <aws_account_id>.dkr.ecr.<region>.amazonaws.com/movie-recommender:latest
```
Run the container:

```bash

docker run -p 8000:8000 <aws_account_id>.dkr.ecr.<region>.amazonaws.com/movie-recommender:latest

```
### Usage

### API Endpoints

GET /: Health check.

POST /recommend: Get movie recommendations.

Example Request
```bash

curl -X POST "http://localhost:8000/recommend" \
-H "Content-Type: application/json" \
-d '{"movie_title": "The Matrix", "user_id": 1}'
```
Example Response
```json

{
  "recommendations": ["Inception", "Interstellar", "The Dark Knight"]
}
```
Kafka Events
User interactions are logged to the user-events Kafka topic. Example event:

```json

{
  "user_id": 1,
  "movie_title": "The Matrix",
  "action": "click",
  "timestamp": "2025-02-05T03:48:37.807Z"
}
```
### Project Structure
```
movie-recommender/
├── api/
│   ├── kafka_client.py       # Kafka producer for user events
│   ├── kafka_consumer.py     # Kafka consumer for online learning
│   └── main.py               # FastAPI backend
├── data/
│   └── movies.csv            # Movie dataset
├── ml/
│   ├── embeddings.pkl        # Precomputed embeddings
│   ├── generate_embeddings.py # Script to generate embeddings
│   └── model.py              # Recommendation model
├── docker-compose.yml        # Docker Compose configuration
├── Dockerfile                # Dockerfile for FastAPI service
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

### Technologies Used
Python: Core programming language.

FastAPI: Backend framework for serving recommendations.

Kafka: Real-time data streaming and online learning.

Docker: Containerization for deployment.

AWS EC2: Cloud hosting for the application.

Sentence Transformers: Semantic embeddings for movie titles.

BM25: Ranking algorithm for recommendations.

Future Enhancements
Hybrid Recommendations: Combine content-based and collaborative filtering.

Vector Database: Use FAISS or Pinecone for faster similarity searches.

Frontend Dashboard: Build a user-friendly interface for exploring recommendations.

Advanced Online Learning: Implement incremental model updates using libraries like River.

Contributors
Agastya

License
This project is licensed under the MIT License. See LICENSE for details.



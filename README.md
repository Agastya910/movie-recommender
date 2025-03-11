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

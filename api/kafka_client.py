from kafka import KafkaProducer
import json
from datetime import datetime

def get_kafka_producer():
    return KafkaProducer(
        bootstrap_servers='kafka:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

def log_user_event(user_id: int, movie_title: str, action: str):
    producer = get_kafka_producer()
    event = {
        "user_id": user_id,
        "movie_title": movie_title,
        "action": action,
        "timestamp": datetime.now().isoformat()
    }
    producer.send('user-events', event)
    producer.flush()

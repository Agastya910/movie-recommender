from kafka import KafkaProducer, KafkaConsumer
import json

# Kafka producer to send messages
def produce_message(topic, message):
    producer = KafkaProducer(
        bootstrap_servers="kafka:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
    producer.send(topic, message)
    producer.flush()

# Kafka consumer to receive messages
def consume_messages(topic):
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers="kafka:9092",
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode("utf-8"))
    )
    for message in consumer:
        yield message.value

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./mistral-movie-recommendation")
tokenizer = AutoTokenizer.from_pretrained("./mistral-movie-recommendation")

# Function to generate recommendations
def generate_recommendation(query):
    # Prepare input for LLM
    input_text = f"Query: {query}\nRecommendation:"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate output
    outputs = model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)
    recommendation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return recommendation

# Example usage
query = "I like sci-fi movies with space exploration."
recommendation = generate_recommendation(query)
print("Recommendation:", recommendation)

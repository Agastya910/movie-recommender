from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# Load synthetic profiles
profiles_df = pd.read_csv("synthetic_profiles_clustered.csv")

# Convert to Hugging Face Dataset format
dataset = Dataset.from_pandas(profiles_df)
train_dataset, val_dataset = dataset.train_test_split(test_size=0.2).values()

# Load Mistral-7B and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize dataset
def tokenize_function(examples):
    inputs = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=512)
    outputs = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=512)
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": outputs["input_ids"]}

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./mistral-movie-recommendation",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,  # Enable mixed precision
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fine-tune
trainer.train()

# Save fine-tuned model
model.save_pretrained("./mistral-movie-recommendation")
tokenizer.save_pretrained("./mistral-movie-recommendation")

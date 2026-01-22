import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset

# Load data
with open("data/finetuning.json") as f:
    data = json.load(f)

# Extract text from nested structure
texts = []
for level in data:
    for chapter in level.get("chapters", []):
        for concept in chapter.get("concepts", []):
            for misconception in concept.get("misconceptions", []):
                texts.append(f"Student: {misconception['student_statement']}\n"
                           f"Belief: {misconception['incorrect_belief']}\n"
                           f"Resolution: {misconception['resolution_insight']}")

# Initialize model and tokenizer
model_name = "distilgpt2"  # Small language model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize dataset
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

dataset = Dataset.from_dict({"text": texts})
tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

# Training configuration
args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=100,
    learning_rate=5e-5,
)

# Train
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
trainer.save_model("./finetuned_model")

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import wandb
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


class MisconceptionDataset:
    """Dataset class for loading and processing misconception data."""

    def __init__(self, json_path):
        self.data = self._load_and_parse(json_path)

    def _load_and_parse(self, json_path):
        """Load JSON and extract student_statement -> incorrect_belief pairs."""
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        pairs = []
        for level in raw_data:
            for chapter in level.get("chapters", []):
                for concept in chapter.get("concepts", []):
                    misconceptions = concept.get("misconceptions", [])
                    for misconception in misconceptions:
                        student_statement = misconception.get(
                            "student_statement", ""
                        ).strip()
                        incorrect_belief = misconception.get(
                            "incorrect_belief", ""
                        ).strip()

                        if student_statement and incorrect_belief:
                            pairs.append(
                                {
                                    "student_statement": student_statement,
                                    "incorrect_belief": incorrect_belief,
                                    "concept": concept.get("concept", "Unknown"),
                                    "topic": chapter.get("topic", "Unknown"),
                                }
                            )

        print(f"Loaded {len(pairs)} misconception pairs from {json_path}")
        return pairs

    def get_formatted_data(self, tokenizer):
        """Format data for training with proper prompt structure."""
        formatted = []
        for item in self.data:
            # Create a clear instruction-following format
            text = (
                f"Student Statement: {item['student_statement']}\n"
                f"Incorrect Belief: {item['incorrect_belief']}{tokenizer.eos_token}"
            )
            formatted.append({"text": text})

        return formatted


def setup_model_and_tokenizer(model_name="LiquidAI/LFM2-700M", device="auto"):
    """Load model and tokenizer from Hugging Face."""
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
    )

    print(f"Model loaded on device: {model.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model, tokenizer


def tokenize_dataset(data, tokenizer, max_length=512):
    """Tokenize the dataset."""

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

    # Convert to Hugging Face dataset
    hf_dataset = HFDataset.from_list(data)
    tokenized = hf_dataset.map(
        tokenize_function, batched=True, remove_columns=hf_dataset.column_names
    )

    return tokenized


def main():
    # Configuration
    MODEL_NAME = "LiquidAI/LFM2-700M"
    DATA_PATH = "data/finetuning.json"
    OUTPUT_DIR = "output/finetuned_model"
    MAX_LENGTH = 512
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 100
    LOGGING_STEPS = 10
    SAVE_STEPS = 200

    # Initialize Weights & Biases
    wandb.init(
        project="ares-slm-1",
        config={
            "model": MODEL_NAME,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
        },
    )

    # Load dataset
    print("=" * 60)
    print("Loading dataset...")
    dataset = MisconceptionDataset(DATA_PATH)

    # Setup model and tokenizer
    print("=" * 60)
    print("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)

    # Format and tokenize data
    print("=" * 60)
    print("Formatting and tokenizing data...")
    formatted_data = dataset.get_formatted_data(tokenizer)
    tokenized_dataset = tokenize_dataset(formatted_data, tokenizer, MAX_LENGTH)

    # Split into train/validation (90/10 split) stratified by chapter
    # First, we need to add chapter/topic labels for stratification

    topic_labels = [item["topic"] for item in dataset.data]

    # Create a mapping to ensure each chapter is represented
    topic_counts = Counter(topic_labels)
    print(f"\nChapter distribution ({len(topic_counts)} unique chapters):")
    for topic, count in sorted(topic_counts.items()):
        print(f"  {topic}: {count} samples")

    # Create stratified split manually
    indices = np.arange(len(tokenized_dataset))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.1, random_state=42, stratify=topic_labels
    )

    train_dataset = tokenized_dataset.select(train_indices.tolist())
    eval_dataset = tokenized_dataset.select(val_indices.tolist())

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")

    # Verify chapter distribution in splits
    val_topics = [topic_labels[i] for i in val_indices]
    print("\nValidation set chapter distribution:")
    val_topic_counts = Counter(val_topics)
    for topic, count in sorted(val_topic_counts.items()):
        print(f"  {topic}: {count} samples")

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling, not masked LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=SAVE_STEPS,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_available(),
        report_to="wandb",
        logging_dir=f"{OUTPUT_DIR}/logs",
        save_strategy="steps",
        remove_unused_columns=False,
        push_to_hub=False,
    )

    # Initialize trainer
    print("=" * 60)
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train the model
    print("=" * 60)
    print("Starting training...")
    print(
        f"Total training steps: {len(train_dataset) * NUM_EPOCHS // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)}"
    )

    trainer.train()

    # Save the final model
    print("=" * 60)
    print("Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Final evaluation
    print("=" * 60)
    print("Final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation loss: {eval_results['eval_loss']:.4f}")

    # Save evaluation results
    results_path = Path("output") / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Evaluation results saved to: {results_path}")

    # Test inference
    print("=" * 60)
    print("Testing inference...")
    test_statement = "Open source models are always worse because they are free."
    test_input = f"Student Statement: {test_statement}\nIncorrect Belief:"

    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=100, temperature=0.7, do_sample=True, top_p=0.9
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nTest Input:\n{test_input}")
    print(f"\nModel Output:\n{generated_text}")

    # Finish WandB run
    wandb.finish()

    print("=" * 60)
    print("Training complete! ✓")
    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

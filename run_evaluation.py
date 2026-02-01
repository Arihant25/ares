import json
import torch
import csv
import time
import os
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

DATA_PATH = "data/finetuning.json"
BASELINE_MODEL_NAME = "LiquidAI/LFM2-700M"
FINETUNED_MODEL_PATH = "output/finetuned_model"
OUTPUT_FILE = "output/evaluation_comparison.json"

def load_one_misconception_per_concept(json_path):
    """
    Loads the dataset and selects exactly one misconception per unique concept.
    Traverses: Level -> Chapter (Topic) -> Concepts -> Misconceptions
    """
    print(f"Reading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_set = []
    
    # The structure is List[Level]
    # Level has "chapters" -> List[Chapter]
    # Chapter has "topic" (string) and "concepts" -> List[Concept]
    # Concept has "misconceptions" -> List[Misconception]
    
    for level in data:
        chapters = level.get("chapters", [])
        for chapter in chapters:
            topic = chapter.get("topic")
            concepts = chapter.get("concepts", [])
            
            for concept in concepts:
                concept_name = concept.get("concept")
                misconceptions = concept.get("misconceptions", [])
                
                # Find first valid misconception for this concept
                for misc in misconceptions:
                    stmt = misc.get("student_statement")
                    belief = misc.get("incorrect_belief")
                    
                    if stmt and belief:
                        test_set.append({
                            "topic": topic,
                            "concept": concept_name,
                            "student_statement": stmt.strip(),
                            "ground_truth_belief": belief.strip()
                        })
                        break
    
    return test_set

def calculate_metrics(candidate, reference, metrics):
    """
    Calculates BLEU, METEOR, ROUGE, and BERTScore.
    metrics: dictionary of loaded evaluate metrics
    """
    scores = {}

    # BLEU
    try:
        # evaluate bleu expects references as list of lists
        b = metrics["bleu"].compute(predictions=[candidate], references=[[reference]])
        scores["bleu"] = b["bleu"]
    except Exception as e:
        scores["bleu"] = 0.0

    # METEOR
    try:
        m = metrics["meteor"].compute(predictions=[candidate], references=[reference])
        scores["meteor"] = m["meteor"]
    except Exception as e:
        scores["meteor"] = 0.0

    # ROUGE
    try:
        r = metrics["rouge"].compute(predictions=[candidate], references=[reference])
        scores["rouge1"] = r["rouge1"]
        scores["rouge2"] = r["rouge2"]
        scores["rougeL"] = r["rougeL"]
    except Exception as e:
        scores["rouge1"] = 0.0
        scores["rouge2"] = 0.0
        scores["rougeL"] = 0.0

    # BERTScore
    try:
        # Using default model (roberta-large)
        bs = metrics["bertscore"].compute(predictions=[candidate], references=[reference], lang="en")
        scores["bertscore_f1"] = bs["f1"][0]
    except Exception as e:
        scores["bertscore_f1"] = 0.0

    return scores


def generate_response(model, tokenizer, statement):
    """
    Generates a response for a single student statement.
    Prompt format:
    Student Statement: {statement}
    Incorrect Belief:
    """
    prompt = f"Student Statement: {statement}\nIncorrect Belief:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generation parameters matching finetune.py inference/test
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False, # Deterministic greedy decoding
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Isolate the generated part
    if full_output.startswith(prompt):
        return full_output[len(prompt):].strip()
    return full_output.strip()

def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 1. Load Test Data
    print("=" * 60)
    test_data = load_one_misconception_per_concept(DATA_PATH)
    print(f"Constructed test set with {len(test_data)} items (one per concept).")
    print("=" * 60)

    # 1.5 Load Metrics
    print("Loading metrics (BLEU, METEOR, ROUGE, BERTScore)...")
    metrics = {
        "bleu": evaluate.load("bleu"),
        "meteor": evaluate.load("meteor"),
        "rouge": evaluate.load("rouge"),
        # Use a smaller model for BERTScore if speed is an issue, but standard is roberta-large
        # Specifying device to avoid issues if cuda available
        "bertscore": evaluate.load("bertscore")
    }
    print("Metrics loaded.")


    # 2. Run Baseline Model
    print(f"Loading Baseline Model: {BASELINE_MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            BASELINE_MODEL_NAME,
            trust_remote_code=True,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        print("Baseline Model loaded.")
        
        print(f"Running inference on {len(test_data)} items with Baseline Model...")
        baseline_bleu_scores = []
        for item in tqdm(test_data, desc="Baseline Inference"):
            start_time = time.time()
            response = generate_response(model, tokenizer, item["student_statement"])
            elapsed = time.time() - start_time
            item["baseline_response"] = response
            item["baseline_time"] = elapsed
            
            scores = calculate_metrics(response, item["ground_truth_belief"], metrics)
            item["baseline_scores"] = scores
            
            # Keep flat bleu for backward compat availability or just use the new one
            baseline_bleu_scores.append(scores["bleu"])
            
        # Cleanup
        del model
        torch.cuda.empty_cache()
        print("Baseline inference complete. Model unloaded.")
        if baseline_bleu_scores:
            print(f"Average Baseline BLEU: {sum(baseline_bleu_scores)/len(baseline_bleu_scores):.4f}")
        
    except Exception as e:
        print(f"Error running baseline model: {e}")
        return

    print("=" * 60)

    # 3. Run Finetuned Model
    print(f"Loading Finetuned Model: {FINETUNED_MODEL_PATH}")
    try:
        # Check if tokenizer exists in finetuned dir, else use baseline tokenizer
        try:
            ft_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH, trust_remote_code=True)
        except:
            print("Using baseline tokenizer for finetuned model.")
            ft_tokenizer = tokenizer # reuse from baseline block if needed, but safer to reload
            ft_tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME, trust_remote_code=True)
        
        if ft_tokenizer.pad_token is None:
            ft_tokenizer.pad_token = ft_tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            FINETUNED_MODEL_PATH,
            trust_remote_code=True,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        print("Finetuned Model loaded.")
        
        print(f"Running inference on {len(test_data)} items with Finetuned Model...")
        finetuned_bleu_scores = []
        for item in tqdm(test_data, desc="Finetuned Inference"):
            start_time = time.time()
            response = generate_response(model, ft_tokenizer, item["student_statement"])
            elapsed = time.time() - start_time
            item["finetuned_response"] = response
            item["finetuned_time"] = elapsed
            
            scores = calculate_metrics(response, item["ground_truth_belief"], metrics)
            item["finetuned_scores"] = scores
            
            finetuned_bleu_scores.append(scores["bleu"])
            
        # Cleanup
        del model
        torch.cuda.empty_cache()
        print("Finetuned inference complete. Model unloaded.")
        if finetuned_bleu_scores:
            print(f"Average Finetuned BLEU: {sum(finetuned_bleu_scores)/len(finetuned_bleu_scores):.4f}")

    except Exception as e:
        print(f"Error running finetuned model: {e}")
    
    # 4. Save Results
    print("=" * 60)
    try:
        # Save detailed JSON
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
        print(f"Evaluation complete. JSON results saved to: {OUTPUT_FILE}")
        
        # Save CSV Summary
        csv_file = OUTPUT_FILE.replace(".json", ".csv")
        fieldnames = [
            "Model", "Topic", "Concept", "Input", "Reference", "Output", 
            "BLEU", "METEOR", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore-F1", "Time(s)"
        ]
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in test_data:
                # Add Baseline Row
                if "baseline_response" in item:
                    scores = item.get("baseline_scores", {})
                    writer.writerow({
                        "Model": "Baseline",
                        "Topic": item.get("topic", ""),
                        "Concept": item.get("concept", ""),
                        "Input": item.get("student_statement", ""),
                        "Reference": item.get("ground_truth_belief", ""),
                        "Output": item.get("baseline_response", ""),
                        "BLEU": f"{scores.get('bleu', 0):.4f}",
                        "METEOR": f"{scores.get('meteor', 0):.4f}",
                        "ROUGE-1": f"{scores.get('rouge1', 0):.4f}",
                        "ROUGE-2": f"{scores.get('rouge2', 0):.4f}",
                        "ROUGE-L": f"{scores.get('rougeL', 0):.4f}",
                        "BERTScore-F1": f"{scores.get('bertscore_f1', 0):.4f}",
                        "Time(s)": f"{item.get('baseline_time', 0):.4f}"
                    })
                
                # Add Finetuned Row
                if "finetuned_response" in item:
                    scores = item.get("finetuned_scores", {})
                    writer.writerow({
                        "Model": "Finetuned",
                        "Topic": item.get("topic", ""),
                        "Concept": item.get("concept", ""),
                        "Input": item.get("student_statement", ""),
                        "Reference": item.get("ground_truth_belief", ""),
                        "Output": item.get("finetuned_response", ""),
                        "BLEU": f"{scores.get('bleu', 0):.4f}",
                        "METEOR": f"{scores.get('meteor', 0):.4f}",
                        "ROUGE-1": f"{scores.get('rouge1', 0):.4f}",
                        "ROUGE-2": f"{scores.get('rouge2', 0):.4f}",
                        "ROUGE-L": f"{scores.get('rougeL', 0):.4f}",
                        "BERTScore-F1": f"{scores.get('bertscore_f1', 0):.4f}",
                        "Time(s)": f"{item.get('finetuned_time', 0):.4f}"
                    })
        print(f"CSV results saved to: {csv_file}")

    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()

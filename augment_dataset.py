import json
import time
from typing import List

from google import genai
from google.genai import types
from pydantic import BaseModel

# === CONFIGURATION ===
API_KEYS = [
    "your",
    "api",
    "keys",
    "here",
]
MODEL_NAMES = ["gemini-3-flash-preview", "gemini-3-pro-preview", "gemini-2.5-flash"]
DATA_FILE = "data/finetuning.json"
TARGET_MISCONCEPTIONS = 8  # 1 existing + 7 new

# === SETUP ===
current_api_key_index = 0
current_model_index = 0
client = genai.Client(api_key=API_KEYS[current_api_key_index])


def cycle_to_next_configuration():
    """Cycle to the next API key/model combination."""
    global current_api_key_index, current_model_index, client

    # Try next model first
    current_model_index += 1
    if current_model_index >= len(MODEL_NAMES):
        # Exhausted all models, move to next API key
        current_model_index = 0
        current_api_key_index += 1
        if current_api_key_index >= len(API_KEYS):
            # Exhausted all API keys, start over
            current_api_key_index = 0
        # Reinitialize client with new API key
        client = genai.Client(api_key=API_KEYS[current_api_key_index])
        print(f"\n    Switched to API key #{current_api_key_index + 1}")

    print(f"    Using model: {MODEL_NAMES[current_model_index]}")
    return MODEL_NAMES[current_model_index]


# === COURSE CURRICULUM CONTEXT ===
CURRICULUM_STRUCTURE = """
Level 1 - Foundations
  Chapters: 4
    ├─ What are LLMs?
    ├─ History & Evolution
    ├─ Basic NLP concepts
    ├─ Everyday applications

Level 2 - Technical Foundations
  Chapters: 5
    ├─ Neural network basics
    ├─ Attention mechanisms
    ├─ Transformer architecture
    ├─ Training data & tokenization
    ├─ Model parameters & scale

Level 3 - Mathematics
  Chapters: 6
    ├─ Linear algebra
    ├─ Probability & statistics
    ├─ Backpropagation
    ├─ Loss functions & optimization
    ├─ Attention math
    ├─ Embeddings & vector spaces

Level 4 - Practical Applications
  Chapters: 5
    ├─ Prompt engineering principles
    ├─ Few-shot learning
    ├─ Chain-of-thought reasoning
    ├─ System prompts & roles
    ├─ Temperature & sampling

Level 5 - Advanced Techniques
  Chapters: 6
    ├─ RAG architecture
    ├─ Vector databases
    ├─ Fine-tuning methods
    ├─ RLHF & alignment
    ├─ Agentic frameworks
    ├─ Multi-agent systems

Level 6 - Ethics & Implications
  Chapters: 6
    ├─ Bias & fairness
    ├─ Hallucinations & reliability
    ├─ Privacy concerns
    ├─ Environmental impact
    ├─ Societal implications
    ├─ Responsible AI development
"""


class Misconception(BaseModel):
    student_statement: str
    incorrect_belief: str
    socratic_sequence: List[str]
    resolution_insight: str
    bloom_level: str


class MisconceptionsList(BaseModel):
    misconceptions: List[Misconception]


def generate_misconceptions(
    topic,
    concept_title,
    existing_misconceptions,
    count_needed,
    level_name,
    all_chapter_misconceptions,
):
    """Generates additional misconceptions using Gemini with retry logic."""

    existing_text = json.dumps(existing_misconceptions, indent=2)
    all_chapter_text = json.dumps(all_chapter_misconceptions, indent=2)

    prompt = f"""
    You are an expert educational content creator for a comprehensive course on Large Language Models.
    
    === COURSE STRUCTURE ===
    {CURRICULUM_STRUCTURE}
    
    === CURRENT CONTEXT ===
    Level: {level_name}
    Chapter/Topic: {topic}
    Concept: {concept_title}
    
    All Misconceptions in this Chapter/Topic (AVOID DUPLICATING ANY OF THESE):
    {all_chapter_text}
    
    Current Misconceptions for THIS Concept:
    {existing_text}
    
    === TASK ===
    Generate {count_needed} NEW, DISTINCT misconceptions students might have about this concept.
    
    Consider:
    - The student's learning level (based on which level of the course they're in)
    - Prerequisites from earlier chapters/levels
    - Common confusions at this stage of learning
    - How this concept relates to other topics in the curriculum
    
    For each misconception, provide:
    1. student_statement: The incorrect statement a student might make
    2. incorrect_belief: The underlying incorrect mental model
    3. socratic_sequence: Exactly 3 guiding questions to lead them to understanding
    4. resolution_insight: The correct understanding they should reach
    5. bloom_level: Appropriate Bloom's Taxonomy level (Understanding, Applying, Analyzing, Evaluating, Creating)
    
    Ensure the content is:
    - High-quality and technically accurate
    - Pedagogically sound with realistic student misconceptions
    - Appropriately scoped for the level in the curriculum
    - Distinct from existing misconceptions
    """

    retry_count = 0
    current_model = MODEL_NAMES[current_model_index]

    while True:
        try:
            response = client.models.generate_content(
                model=current_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=MisconceptionsList,
                ),
            )

            if response.parsed:
                return response.parsed.misconceptions
            else:
                print(f"\n    Warning: No parsed content returned for {concept_title}.")
                current_model = cycle_to_next_configuration()
                time.sleep(5)
                retry_count += 1

        except Exception as e:
            print(f"\n    Error generating for {concept_title}: {e}")
            current_model = cycle_to_next_configuration()
            print(f"    Retrying... (attempt #{retry_count + 1})")
            time.sleep(5)
            retry_count += 1


def main():
    print(f"Loading {DATA_FILE}...")
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found.")
        return

    print(
        f"Starting processing with API key #{current_api_key_index + 1} and model {MODEL_NAMES[current_model_index]}..."
    )

    modified_count = 0
    concepts_processed = 0

    for level_idx, level_data in enumerate(data):
        chapters = level_data.get("chapters", [])
        level_name = f"Level {level_data.get('level')} - {level_data.get('title')}"
        print(f"\nProcessing {level_name} ({len(chapters)} chapters)")

        for chapter in chapters:
            topic = chapter.get("topic")
            concepts = chapter.get("concepts", [])
            print(f"  > Chapter: {topic} ({len(concepts)} concepts)")

            # Collect all misconceptions from this chapter for context
            all_chapter_misconceptions = []
            for c in concepts:
                for m in c.get("misconceptions", []):
                    all_chapter_misconceptions.append(
                        {
                            "concept": c.get("concept"),
                            "student_statement": m.get("student_statement"),
                            "incorrect_belief": m.get("incorrect_belief"),
                        }
                    )

            for concept_data in concepts:
                concept_title = concept_data.get("concept")
                current_misconceptions = concept_data.get("misconceptions", [])

                count_needed = TARGET_MISCONCEPTIONS - len(current_misconceptions)

                if count_needed > 0:
                    print(
                        f"    - Generating {count_needed} misconceptions for '{concept_title}'...",
                        end="",
                        flush=True,
                    )

                    new_items = generate_misconceptions(
                        topic,
                        concept_title,
                        current_misconceptions,
                        count_needed,
                        level_name,
                        all_chapter_misconceptions,
                    )

                    # generate_misconceptions now has infinite retry, so this will always succeed
                    # Convert Pydantic models to dicts
                    new_dicts = [item.model_dump() for item in new_items]
                    concept_data["misconceptions"].extend(
                        new_dicts[:count_needed]
                    )  # Ensure we don't overfill
                    modified_count += 1
                    print(f" Done. (Total: {len(concept_data['misconceptions'])})")

                    # Update the chapter context with newly generated misconceptions
                    for new_dict in new_dicts[:count_needed]:
                        all_chapter_misconceptions.append(
                            {
                                "concept": concept_title,
                                "student_statement": new_dict.get("student_statement"),
                                "incorrect_belief": new_dict.get("incorrect_belief"),
                            }
                        )

                    # Rate limit kindness
                    time.sleep(1)
                else:
                    print(
                        f"    ✓ Skipping '{concept_title}' (already has {len(current_misconceptions)} misconceptions)"
                    )
                    pass

                concepts_processed += 1

                # Save after every concept to preserve progress
                with open(DATA_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(f"    [Saved progress - {concepts_processed} concepts processed]")

    print(f"\nCompleted! Modified concepts: {modified_count}")


if __name__ == "__main__":
    main()

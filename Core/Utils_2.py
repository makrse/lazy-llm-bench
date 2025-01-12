from enum import Enum
from typing import Union, Optional, List, Dict, Pattern
import re
from pydantic import ValidationError
from deepeval.benchmarks.big_bench_hard.big_bench_hard import (
    bbh_confinement_statements_dict,
    BigBenchHardTask
)

# Type 1 [ARC; HellaSwag; MMLU]

def generate_type_1(model, prompt: str, **kwargs):
    """Generate a response from the model."""
    try:
        # Generate the response
        response = model(prompt=prompt, **kwargs)

        if "choices" in response:
            generated_text = response["choices"][0]["text"].strip()
        elif "generated_text" in response:
            generated_text = response["generated_text"].strip()
        else:
            print(f"Unexpected model output: {response}")
            generated_text = ""

        # Check for a valid answer format (A, B, C, D)
        match = re.search(r"\b[A-D]\b", generated_text)  # Look for a valid answer ('A', 'B', 'C', or 'D')
        if match:
            return match.group(0)  # Return the valid answer
        else:
            print(f"Invalid answer format detected: {generated_text}")
            return None  # Return None for invalid answers

    except Exception as e:
        # Handle any unexpected errors
        print(f"Error during generation: {e}")
        return None  # Return None in case of errors

# Type 2 [BigBenchHard; ]

def generate_type_2(model, prompt: str, task: Union[str, List[str]] = None, **kwargs): 
    """Generate a response for BigBenchHard tasks with simplified logic and output formatting based on task."""
    from BigBenchHard_Benchmark import BB_bench_tasks
    original_prompt = f"\n{prompt}"

    task_keywords = {
        BigBenchHardTask.BOOLEAN_EXPRESSIONS: ["boolean", "expression", "true", "false"],
        BigBenchHardTask.WORD_SORTING: ["sorting", "order", "sequence", "word"],
        BigBenchHardTask.CAUSAL_JUDGEMENT: ["cause", "effect", "judgement", "reasoning"],
        BigBenchHardTask.DATE_UNDERSTANDING: [], # Need to add some key words for each tasks to able to match the correct task its instruction as a method to better handle.
        BigBenchHardTask.DISAMBIGUATION_QA: [],
        BigBenchHardTask.DYCK_LANGUAGES: [],
        BigBenchHardTask.FORMAL_FALLACIES: [],
        BigBenchHardTask.GEOMETRIC_SHAPES: [],
        BigBenchHardTask.HYPERBATON: [],
        BigBenchHardTask.LOGICAL_DEDUCTION_THREE_OBJECTS: [],
        BigBenchHardTask.LOGICAL_DEDUCTION_FIVE_OBJECTS: [],
        BigBenchHardTask.LOGICAL_DEDUCTION_SEVEN_OBJECTS: [],
        BigBenchHardTask.MOVIE_RECOMMENDATION: [],
        BigBenchHardTask.MULTISTEP_ARITHMETIC_TWO: [],
        BigBenchHardTask.NAVIGATE: [],
        BigBenchHardTask.OBJECT_COUNTING: [],
        BigBenchHardTask.PENGUINS_IN_A_TABLE: [],
        BigBenchHardTask.REASONING_ABOUT_COLORED_OBJECTS: [],
        BigBenchHardTask.RUIN_NAMES: [],
        BigBenchHardTask.SALIENT_TRANSLATION_ERROR_DETECTION: [],
        BigBenchHardTask.SNARKS: [],
        BigBenchHardTask.SPORTS_UNDERSTANDING: [],
        BigBenchHardTask.TEMPORAL_SEQUENCES: [],
        BigBenchHardTask.TRACKING_SHUFFLED_OBJECTS_THREE_OBJECTS: [],
        BigBenchHardTask.TRACKING_SHUFFLED_OBJECTS_FIVE_OBJECTS: [],
        BigBenchHardTask.TRACKING_SHUFFLED_OBJECTS_SEVEN_OBJECTS: [],
        BigBenchHardTask.WEB_OF_LIES: [],
    }

    task_instructions = bbh_confinement_statements_dict

    if task is None:
        task = BB_bench_tasks

    matched_tasks = []
    for task_type, keywords in task_keywords.items():
        if any(keyword.lower() in prompt.lower() for keyword in keywords):
            matched_tasks.append(task_type)

    if matched_tasks:
        task = matched_tasks[0] 
        instruction = task_instructions.get(task, "No instructions available for this task.")
        
        #==========This line need to improve for better instruccion over each task expected output format=====#
        new_prompt = f"{original_prompt}\n\nInstruction: {instruction}\n\nPlease respond ONLY with the result as specified in the instruction. DO NOT include any extra information, explanations, or context. The result should match the format exactly, and ONLY the result should be output."
        #==========Maybe its the model that its not able to follow the instructions correctly=================#
    
    else:
        print("\nError: No valid task found from prompt keywords.")

    clear_prompt = re.sub(r'\s+', ' ', new_prompt)  # Clean extra spaces
    print(f"\nUpdated prompt with task-specific instructions: {clear_prompt}")

    if "schema" in kwargs:
        kwargs.pop("schema")

    response = model(prompt=clear_prompt, **kwargs)
    print(f"\nresponse: {response}")
    generated_text = extract_generated_text(response)
    print(f"\ngenerated_text: {generated_text}")
    return generated_text

def extract_generated_text(response):
    """Extract generated text from model response."""
    if isinstance(response, str):
        return response.strip()
    elif isinstance(response, dict):
        if "choices" in response and isinstance(response["choices"], list):
            return response["choices"][0].get("text", "").strip()
        elif "generated_text" in response:
            return response["generated_text"].strip()
    raise ValueError("Unexpected response format")


# Type 3 [Drop_Bench; ]
def generate_type_3(model, prompt: str, schema=None) -> Dict:
    """
    Generate predictions for a single prompt. 
    Ensures the response adheres to the schema if provided.
    """
    result = model(prompt)["choices"][0]["text"].strip()
    # Validate schema (if provided)
    if schema:
        try:
            return schema.parse_obj({"answer": result})
        except ValidationError as e:
            print(f"Schema validation failed: {e}")
    return {"answer": result}


import re


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


# Type 2 [BigBenchHard; ***]
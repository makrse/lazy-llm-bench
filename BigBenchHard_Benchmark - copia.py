import os
import re
from llama_cpp import Llama
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks.big_bench_hard.big_bench_hard import (
    BigBenchHard,
    BigBenchHardTask,
    bbh_confinement_statements_dict
)

class GGUFModel(DeepEvalBaseLLM):
    def load_model(
            self,
            model_path: str, 
            context_n_ctxlength: int = 1024, 
            temperature: float = 0, 
            **kwargs
        ):
        """Loads a GGUF model using llama.cpp."""
        return Llama(
            model_path=model_path,
            n_ctx=context_n_ctxlength,
            temperature=temperature,
            **kwargs
        )
        

    def generate(self, prompt: str, schema=None, task=None, **kwargs):
        task = [BigBenchHardTask.WORD_SORTING, BigBenchHardTask.BOOLEAN_EXPRESSIONS, BigBenchHardTask.CAUSAL_JUDGEMENT]  # Remember to modify this be same as the task you want to run.

        if not task:
            print("Task is required but not provided.")
            return None
        if isinstance(task, list):
            confinement_instructions = [bbh_confinement_statements_dict.get(t) for t in task]
        else:
            confinement_instructions = [bbh_confinement_statements_dict.get(task)]
        if not any(confinement_instructions):
            print(f"No confinement instructions found for provided task(s): {task}")
            return None
        try:
            # Generate the response
            response = self.model(prompt=prompt, **kwargs)
            if "choices" in response:
                generated_text = response["choices"][0]["text"].strip()
            elif "generated_text" in response:
                generated_text = response["generated_text"].strip()
            else:
                print(f"Unexpected model output: {response}")
                return None

            # Format the output for the first valid task instruction
            for instruction in confinement_instructions:
                if instruction:
                    formatted_output = self._format_output(generated_text, instruction)
                    if formatted_output is not None:
                        return formatted_output

            print(f"Failed to format output for task(s): {task}. Raw output: {generated_text}")
            return None
        except Exception as e:
            print(f"Error during generation: {e}")
            return None
        
    def _format_output(self, generated_text: str, instruction: str):
        """
        Format the model output based on the task-specific confinement instruction.

        Args:
            generated_text (str): The raw output from the model.
            instruction (str): The confinement instruction for the task.

        Returns:
            str: The formatted output, or None if formatting fails.
        """
        try:
            # Map instruction keywords to expected output patterns
            if "Output '(A)'" in instruction:
                match = re.search(r"\b[A-G]\b", generated_text)
            elif "Output 'Yes' or 'No'" in instruction:
                match = re.search(r"\b(Yes|No)\b", generated_text, re.IGNORECASE)
            elif "Output 'True' or 'False'" in instruction:
                match = re.search(r"\b(True|False)\b", generated_text, re.IGNORECASE)
            elif "Output the numerical answer" in instruction:
                match = re.search(r"\b\d+(\.\d+)?\b", generated_text)  
            elif "Output 'valid' or 'invalid'" in instruction:
                match = re.search(r"\b(valid|invalid)\b", generated_text, re.IGNORECASE)
            elif "sequence of parentheses characters" in instruction:
                match = re.search(r"^[\(\)\s]+$", generated_text.strip())
            elif "sequence of words" in instruction:                            ######## Failing, need a fix
                match = re.search(r"^[a-zA-Z\s]+$", generated_text.strip())
            else:
                print(f"Unknown formatting instruction: {instruction}")
                return None

            if match:
                return match.group(0).strip()
            else:
                print(f"Invalid output format detected: {generated_text}")
                return None

        except Exception as e:
            print(f"Error during output formatting: {e}")
            return None

    def get_model_name(self, *args, **kwargs) -> str:
        return "GGUF"
    
    def a_generate(self, *args, **kwargs):
        return super().a_generate(*args, **kwargs)

    def close_model(self):
        if hasattr(self, 'model') and self.model is not None:
            print("Closing model...")
            self.model.close()
            self.model = None

class GGUFModelLoader:
    def __init__(
        self,
        model_path: str,
        context_length: int = 1024,
        temperature: float = 0,
        n_predict: int = -1,
        cpu_threads: int = 4,
        n_gpu_layers: int = 0,
        keep_model_in_ram: bool = False, 
        use_flash_attention: bool = True,
        use_fp16: bool = True
    ):
        self.model_path = model_path
        self.context_length = context_length
        self.temperature = temperature
        self.n_predict = n_predict
        self.cpu_threads = cpu_threads
        self.n_gpu_layers = n_gpu_layers
        self.keep_model_in_ram = keep_model_in_ram
        self.use_flash_attention = use_flash_attention
        self.use_fp16 = use_fp16
        self.model = None
        self.tokenizer = None
        self.chat_history = []  # Initialize chat history
        os.environ["OMP_NUM_THREADS"] = str(self.cpu_threads)

    def load(self):
        return GGUFModel(
            model_path=self.model_path,
            context_n_ctxlength=self.context_length,  # Explicitly map context_length to n_ctx
            temperature=self.temperature,
            n_predict=self.n_predict,
            n_threads=self.cpu_threads,
            use_mlock=self.keep_model_in_ram,
            use_mmap=not self.keep_model_in_ram,
            n_gpu_layers=self.n_gpu_layers,
            use_flash_attention=self.use_flash_attention,
            use_fp16=self.use_fp16,
            verbose=False
        )

if __name__ == "__main__":
    model_path = os.path.join("name_of_model.gguf")
    
    gguf_loader = GGUFModelLoader(
        model_path=model_path, 
        context_length=0, # Actual token lengh of the model you want to asign so the model memorize, mostly they tell you how much tokens, it could consume ram
        temperature=0, # 0 rigid , more than "1" creative but allucinate.
        n_predict=-1, # This does not take in count, depends, its for how many tokens you want to give to generate an answer, but since its a,b,c,d answers...
        cpu_threads=0, # how many cpu cores to give it, 
        n_gpu_layers=0, # how many gpu layers/cores to give it 0-60?, give more , more fast response.
        keep_model_in_ram=True, # Put or not the llm model into the ram, despite being false or true it will also load on the gpu, you could try false or true to see difference.
        use_flash_attention=True, # Make fast , its the "thing"
        use_fp16=True  # Enable FP16 , being said its for consumer gpus, if you had other type that allows bf16, you could tweak it, for me fp16 its fast over fp32, almost same acc.
    )
    model = gguf_loader.load()

    task = [BigBenchHardTask.WORD_SORTING, BigBenchHardTask.BOOLEAN_EXPRESSIONS, BigBenchHardTask.CAUSAL_JUDGEMENT]   # check their task, probably some not run well, if the prediction its empty, its probably the model not being trained well.

    benchmark = BigBenchHard(
        tasks=task,
        n_shots=0,   # Check their page , 0 to zero-shot.
        enable_cot=True, 
        verbose_mode=False # log stuff in console.
    )

    benchmark.evaluate(model)
    print(f"BigBenchHard Benchmark Results: {benchmark.overall_score}")

    model.close_model()
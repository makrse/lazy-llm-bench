import os
import re

from llama_cpp import Llama
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask

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

    def generate(self, prompt: str, schema=None, **kwargs):
        """Generate a response from the model."""
        try:
            # Generate the response
            response = self.model(prompt=prompt, **kwargs)

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


    def get_model_name(self, *args, **kwargs) -> str:
        return "GGUF"
    
    def a_generate(self, *args, **kwargs):
        return super().a_generate(*args, **kwargs)

    def close_model(self):
        if self.model is not None:
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
            verbose=False # log stuffs on console. sometimes useful see if gpu works or not and other loading model data.
        )
    
# Do not modify above if you dont know what it does.
# You can modify below to test the code.

if __name__ == "__main__":
    model_path = os.path.join("llama2_13b.gguf")
    
    gguf_loader = GGUFModelLoader(
        model_path=model_path, 
        context_length=4096,
        temperature=0,
        n_predict=-1, 
        cpu_threads=4, 
        n_gpu_layers=40, 
        keep_model_in_ram=True, 
        use_flash_attention=True,
        use_fp16=True  # Enable FP16,
    )
    model = gguf_loader.load()

    benchmark = MMLU(
        tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY],
        n_shots=0
    )

    benchmark.evaluate(model)
    print(f"MMLU Benchmark Results: {benchmark.overall_score}")

    model.close_model()
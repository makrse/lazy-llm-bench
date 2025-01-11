import os
from llama_cpp import Llama
from deepeval.models.base_model import DeepEvalBaseLLM
from .Utils_2 import generate_type_1
# Setting up Core loading llma.cpp base stuff
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
        return generate_type_1(self.model, prompt, **kwargs)
    
    def get_model_name(self, *args, **kwargs) -> str:
        return "GGUF"
    
    def a_generate(self, *args, **kwargs):
        return super().a_generate(*args, **kwargs)
    
    def close_model(self):
        if self.model is not None:
            self.model.close()
            self.model = None

# Setting up loader kwargs and parameters

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
        use_fp16: bool = True,
        use_fp32: bool = False,
        verbose: bool = False
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
        self.use_fp32= use_fp32
        self.verbose = verbose
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
            use_fp32=self.use_fp32,
            verbose=self.verbose
        )
import os
import re
from llama_cpp import Llama
from pydantic import ValidationError
from typing import Dict, List

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks import DROP
from deepeval.benchmarks.tasks import DROPTask
from deepeval.benchmarks.drop.drop import DROPDateSchema,DROPNumberSchema,DROPStringSchema





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
        

    def generate(self, prompt: str, schema=None) -> Dict:
        """
        Generate predictions for a single prompt. 
        Ensures the response adheres to the schema if provided.
        """
        result = self.model(prompt)["choices"][0]["text"].strip()
        # Validate schema (if provided)
        if schema:
            try:
                return schema.parse_obj({"answer": result})
            except ValidationError as e:
                print(f"Schema validation failed: {e}")
        return {"answer": result}

    def batch_generate(self, prompts: List[str], schemas: List = None) -> List[Dict]:
        """
        Generate predictions for multiple prompts in a batch. 
        Ensures the responses adhere to their respective schemas.
        """
        results = [self.model(prompt)["choices"][0]["text"].strip() for prompt in prompts]
        if schemas:
            validated_results = []
            for res, schema in zip(results, schemas):
                try:
                    validated_results.append(schema.parse_obj({"answer": res}))
                except ValidationError:
                    validated_results.append({"answer": res})
            return validated_results
        return [{"answer": res} for res in results]

    
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

    benchmark = DROP(
        tasks=[DROPTask.HISTORY_1002, DROPTask.NFL_649], # check their task, current not doing great, need further tweaks, because dataset expected output against the model prediction not equals 
        n_shots=0, # Check their page , 0 to zero-shot.
        verbose_mode=True # log stuff in console.
    )


    benchmark.evaluate(model)
    print(f"DROP Benchmark Results: {benchmark.overall_score}")

    model.close_model()
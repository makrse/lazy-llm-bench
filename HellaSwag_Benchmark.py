import os
from Core.Core_1 import GGUFModelLoader
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks import HellaSwag
from deepeval.benchmarks.tasks import HellaSwagTask


if __name__ == "__main__":
    model_path = os.path.join("name_of_llm.gguf") # self descriptive.
    
    gguf_loader = GGUFModelLoader(
        model_path=model_path, 
        context_length=0, # Actual token lengh of the model you want to asign so the model memorize, mostly they tell you how much tokens, it could consume ram
        temperature=0, # 0 rigid , more than "1" creative but allucinate.
        n_predict=-1, # This does not take in count, depends, its for how many tokens you want to give to generate an answer, but since its a,b,c,d answers...
        cpu_threads=0, # how many cpu cores to give it", 
        n_gpu_layers=0, # how many gpu layers/cores to give it 0-60?, give more , more fast response.
        keep_model_in_ram=True, # Put or not the llm model into the ram, despite being false or true it will also load on the gpu, you could try false or true to see difference.
        use_flash_attention=True, # Make fast , its the "thing"
        use_fp16=True,  # Enable FP16 , being said its for consumer gpus, if you had other type that allows bf16, you could tweak it, for me fp16 its fast over fp32, almost same acc.
        use_fp32=False, # Only need to choose 1 either fp16 or fp32
        verbose=False, # In any case you want to see the loading process set to True.
    )
    model = gguf_loader.load()

    benchmark = HellaSwag( # check their task list, some are useful, some are not, like Health task only has 4 entries, gives good score making not that "good" for overal score if rigid
        tasks=[HellaSwagTask.COMPUTERS_AND_ELECTRONICS, HellaSwagTask.INSTALLING_CARPET, HellaSwagTask.EDUCATION_AND_COMMUNICATIONS,
               HellaSwagTask.WORK_WORLD, HellaSwagTask.FINANCE_AND_BUSINESS, 
               ],
        n_shots=0, # being 0 called zero shots, they said default being 10, no more than 15.
        verbose_mode=True # log stuffs on console.
    )

    # Evaluate the model on ARC
    benchmark.evaluate(model)
    print(f"HellaSwag Benchmark Results: {benchmark.overall_score}")

    model.close_model()

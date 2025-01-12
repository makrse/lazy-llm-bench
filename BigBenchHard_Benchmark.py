import os
from Core.Core_2 import GGUFModelLoader
from deepeval.benchmarks.big_bench_hard.big_bench_hard import (
    BigBenchHard,
    BigBenchHardTask,
)

BB_bench_tasks = [BigBenchHardTask.BOOLEAN_EXPRESSIONS, BigBenchHardTask.WORD_SORTING, BigBenchHardTask.CAUSAL_JUDGEMENT]  

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
        use_fp16=True,  # Enable FP16 , being said its for consumer gpus, if you had other type that allows bf16, you could tweak it, for me fp16 its fast over fp32, almost same acc.
        use_fp32=False,
        verbose=True
    )
    model = gguf_loader.load()

    benchmark = BigBenchHard(
        tasks=BB_bench_tasks,
        n_shots=0,   # Check their page , 0 to zero-shot.
        enable_cot=True, 
        verbose_mode=False, # log stuff in console.
        verbose_mode=True,
        n_problems_per_task=1, # number of tasks per task | I recommend try only 1 to 3 because the prompting of the utils_2 over this method need improve or the model bad on following instructions
    )

    benchmark.evaluate(model)
    print(f"BigBenchHard Benchmark Results: {benchmark.overall_score}")

    model.close_model()

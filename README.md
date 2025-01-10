# Hi little lazy
This is my first mini project as hobby.
This project start because for regular users its very overhealming to get to know about how to bench llm without going through many videos,docs,etc.

Main focused on using gguf models and gpu(cuda) at moment.

# How to use it:

1. Make an envirioment using anaconda[conda] or python venv, and choose:
   - python version 3.10(ideal)
   - python version 3.10.8(current im using) (I didnt test other version of py)
2. Install llama.cpp (python) 
   - [Cannot go inside every error that raises when installing that, I manage to get mine working, every system has their own errors probably.]
>
3. Download the DeepEval repository files, only the main carpet and put it into an empty carpet.
>
4. pip install -U deepeval==2.1.2
>
5. Use my repo and put it on the main carpet, replace the files from mine to theirs.
>
6. Open the [bench type].py you want to run and I suggest to put the gguf model into the main carpet, modify the parameters if you want to fit your use case.
>
7. Run it.
    - py bench_type.py




# Final notes:
Thanks to DeepEval and llama.cpp team for their inmense work.
I will update to add some other bench types when I get to know how useful its, since theres so many,
attempt to do all of it could consume time, electricty and probably not very well formated bench dataset, they all have their cons.

If you use my repo and feels its useful, please give me feedback and dont forget to mention it the repo, It can make me feel I could do more stuffs to share.\
if you have some request you could give me a tickle, I will look at it on my free time.

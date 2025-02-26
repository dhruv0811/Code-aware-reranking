#!/bin/bash
#SBATCH --job-name=generation
#SBATCH --partition=general
#SBATCH --time=48:00:00 

#SBATCH --mem=32000
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4

#SBATCH --output=/home/gganeshl/Code-aware-reranking/slurm/logs/output-%j.log
#SBATCH --error=/home/gganeshl/Code-aware-reranking/slurm/logs/error-%j.out

#SBATCH --mail-type=END
#SBATCH --mail-user=gganeshl@andrew.cmu.edu

export HF_API_KEY=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

source /home/gganeshl/miniconda3/etc/profile.d/conda.sh
conda activate rag

cd /home/gganeshl/Code-aware-reranking

# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_variables_k5.json --k_values 1,5,10 --model_name meta-llama/Llama-3.1-8B-Instruct > var.txt
# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_functions_k5.json --k_values 1,5,10 --model_name meta-llama/Llama-3.1-8B-Instruct > fun.txt
# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_both_k5.json --k_values 0,1,5,10 --model_name meta-llama/Llama-3.1-8B-Instruct > both.txt


# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_none_k5.json --k_values 1,5,10 --model_name Qwen/Qwen2.5-Coder-7B-Instruct > none.txt
# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_docstring_k5.json --k_values 1,5,10 --model_name Qwen/Qwen2.5-Coder-7B-Instruct > doc.txt
# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_variables_k5.json --k_values 1,5,10 --model_name Qwen/Qwen2.5-Coder-7B-Instruct > var.txt
# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_functions_k5.json --k_values 1,5,10 --model_name Qwen/Qwen2.5-Coder-7B-Instruct > fun.txt
# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_both_k5.json --k_values 0,1,5,10 --model_name Qwen/Qwen2.5-Coder-7B-Instruct > both.txt

## bigcode/starcoder2-7b
python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_none_k5.json --k_values 1,5,10 --model_name bigcode/starcoder2-7b > none.txt
# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_docstring_k5.json --k_values 1,5,10 --model_name bigcode/starcoder2-7b > doc.txt
# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_variables_k5.json --k_values 1,5,10 --model_name bigcode/starcoder2-7b > var.txt
# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_functions_k5.json --k_values 1,5,10 --model_name bigcode/starcoder2-7b > fun.txt
# python src/Generation.py --json_input /home/gganeshl/Code-aware-reranking/results/humaneval_best_saved/retrieved_docs/docs_openai_humaneval_Llama-3.1-8B-Instruct_GIST-large-Embedding-v0_both_k5.json --k_values 0,1,5,10 --model_name bigcode/starcoder2-7b > both.txt

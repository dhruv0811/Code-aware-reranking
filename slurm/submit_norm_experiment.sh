#!/bin/bash
#SBATCH --job-name=norm_experiment
#SBATCH --partition=general
#SBATCH --time=48:00:00 

#SBATCH --mem=32000
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4

#SBATCH --output=/home/dhruvgu2/CodeRAG-reranking/slurm/logs/norm_experiment-output-%j.log
#SBATCH --error=/home/dhruvgu2/CodeRAG-reranking/slurm/logs/norm_experiment-error-%j.out

#SBATCH --mail-type=END
#SBATCH --mail-user=dhruvgu2@andrew.cmu.edu

export HF_API_KEY=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

source /home/dhruvgu2/miniconda3/etc/profile.d/conda.sh
conda activate rag

cd /home/dhruvgu2/CodeRAG-reranking

python3 src/analysis/run_normalization_experiment.py
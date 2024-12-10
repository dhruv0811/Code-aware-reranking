#!/bin/bash
#SBATCH --job-name=humaneval_pseudocode
#SBATCH --partition=general
#SBATCH --time=48:00:00 

#SBATCH --mem=32000
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=4

#SBATCH --output=/home/gganeshl/CodeRAG-reranking/slurm/logs/output-%j.log
#SBATCH --error=/home/gganeshl/CodeRAG-reranking/slurm/logs/error-%j.out

#SBATCH --mail-type=END
#SBATCH --mail-user=gganeshl@andrew.cmu.edu

export HF_API_KEY=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

source /home/gganeshl/miniconda3/etc/profile.d/conda.sh
conda activate coderag

cd /home/gganeshl/CodeRAG-reranking

python src/PseudocodeExperiments.py
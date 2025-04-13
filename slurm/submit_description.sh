#!/bin/bash
#SBATCH --job-name=description_pseudo_reranking
#SBATCH --partition=general
#SBATCH --time=48:00:00 

#SBATCH --mem=32000
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4

#SBATCH --output=/home/gganeshl/Code-aware-reranking/slurm/logs/outputDesc-%j.log
#SBATCH --error=/home/gganeshl/Code-aware-reranking/slurm/logs/errorDesc-%j.out

#SBATCH --mail-type=END
#SBATCH --mail-user=gganeshl@andrew.cmu.edu

export HF_API_KEY=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

source /home/gganeshl/miniconda3/etc/profile.d/conda.sh
conda activate rag

cd /home/gganeshl/Code-aware-reranking

python src/DescriptionEvaluationExperiments.py
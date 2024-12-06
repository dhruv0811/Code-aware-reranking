#!/bin/bash
#SBATCH --job-name=ds1000_test
#SBATCH --partition=general
#SBATCH --time=24:00:00 

#SBATCH --mem=32000
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4

#SBATCH --output=/home/dhruvgu2/CodeRAG-reranking/slurm/logs/output-%j.log
#SBATCH --error=/home/dhruvgu2//CodeRAG-reranking/slurm/logs/error-%j.out

# # SBATCH --mail-type=END
# # SBATCH --mail-user=dhruvgu2@andrew.cmu.edu

export HF_API_KEY=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

source /home/dhruvgu2/miniconda3/etc/profile.d/conda.sh
conda activate rag

cd /home/dhruvgu2/CodeRAG-reranking

python src/pipeline_test_ds_1000.py
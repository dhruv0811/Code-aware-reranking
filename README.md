# Code-Aware Reranking for Retrieval in Low Documentation Settings

This project implements code-aware reranking and pseudocode generation approaches to improve code retrieval in low-documentation environments. It addresses the challenges of retrieving relevant code when documentation is limited or missing.

## Features

- Code-aware reranking using large language models
- Pseudocode generation for improved retrieval
- Support for multiple normalization techniques:
  - Docstring removal
  - Function name normalization
  - Variable name normalization
  - Combined normalization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/code-aware-reranking.git
cd code-aware-reranking
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Environment Setup

Set your Hugging Face API key:
```bash
export HF_API_KEY="your_api_key_here"
```

## Usage

### Running Reranking Experiments

```python
from RerankingExperiments import run_experiments

experiment_id = run_experiments(
    output_dir="results/fixed_corpus_humaneval_reranker",
    num_samples=None
)
```

### Running Pseudocode Generation Experiments

```python
from PseudocodeExperiments import run_experiments

experiment_id = run_experiments(
    output_dir="results/humaneval_pseudocode",
    num_samples=None
)
```

## Configuration

The project supports multiple model configurations:

### LLM Models
- meta-llama/Llama-3.1-70B-Instruct
- meta-llama/Llama-3.1-8B-Instruct
- mistralai/Mixtral-8x7B-Instruct-v0.1

### Embedding Models
- avsolatorio/GIST-large-Embedding-v0
- avsolatorio/GIST-Embedding-v0
- sentence-transformers/all-mpnet-base-v2
- flax-sentence-embeddings/st-codesearch-distilroberta-base


## Project Structure

```
├── Corpus.py                     # Corpus handling and normalization
├── Reranking.py                  # Code-aware reranking implementation
├── RerankingExperiments.py       # Reranking experiment runner
├── Pseudocode.py                 # Pseudocode generation implementation
├── PseudocodeExperiments.py      # Pseudocode experiment runner
└── results/                      # Experiment results directory
```

## Results

Experiment results are saved in CSV and JSON formats, including:
- Baseline recall scores
- Reranked recall scores
- Performance metrics for different normalization techniques
- Summary statistics and best configurations


## Contributors

- Dhruv Gupta - dhruvgu2@andrew.cmu.edu
- Gayathri Ganesh Lakshmy - gganeshl@andrew.cmu.edu
- Daniel Chechelnitsky - dchechel@andrew.cmu.edu
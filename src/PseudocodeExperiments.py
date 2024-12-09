import os
import json
from datetime import datetime
from itertools import product
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from Corpus import ProgrammingSolutionsCorpus
from Pseudocode import HumanEvalPseudoRetrieval

# Configuration constants
LLM_MODELS = [
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]

EMBEDDING_MODELS = [
    "avsolatorio/GIST-large-Embedding-v0",
    "avsolatorio/GIST-Embedding-v0",
    "sentence-transformers/all-mpnet-base-v2",
    "jinaai/jina-embeddings-v2-base-code",
    "flax-sentence-embeddings/st-codesearch-distilroberta-base",
]

NORMALIZATION_TYPES = [
    "none",
    "docstring",
    "variables",
    "functions",
    "both"
]


IS_PSEUDO = [True, False]

# Eval Recall@K Values
K_VALUES = [1, 5, 10, 25, 50, 100]

def generate_experiment_id() -> str:
    """Generate a unique identifier for the experiment run."""
    timestamp = datetime.now().strftime("redo3_%Y%m%d_%H%M%S")
    return f"experiment_{timestamp}"

def save_results(results: List[Dict], experiment_id: str, output_dir: Path):
    """Save experiment results to CSV and JSON with summary statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save as CSV
    csv_path = output_dir / f"{experiment_id}_results.csv"
    df.to_csv(csv_path, index=False)
    
    # Save as JSON with pretty printing
    json_path = output_dir / f"{experiment_id}_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    
    # Generate summary statistics
    summary_path = output_dir / f"{experiment_id}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=== Experiment Summary ===\n\n")
        
        print("Dataframe: ", df)

        # Best configurations for each k value
        f.write("Best Configurations by Recall@K:\n")
        for k in K_VALUES:
            df_filtered = df[~df[f'pseudo_recall@{k}'].isna()]
            if not df_filtered.empty:
                best_idx = df_filtered[f'pseudo_recall@{k}'].idxmax()
                best_config = df_filtered.loc[best_idx]
                f.write(f"\nBest for Recall@{k} = {best_config[f'pseudo_recall@{k}']:.3f}:\n")
                f.write(f"- LLM: {best_config['llm_model']}\n")
                f.write(f"- Embeddings: {best_config['embedding_model']}\n")
                f.write(f"- Normalization: {best_config['normalization_type']}\n")
                f.write(f"- Pseudo: {best_config['is_pseudo']}\n")
        
        # Average improvement over baseline
        improvements = []
        for k in K_VALUES:
            baseline_col = f'baseline_recall@{k}'
            reranked_col = f'pseudo_recall@{k}'
            if baseline_col in df.columns and reranked_col in df.columns:
                improvement = (df[reranked_col] - df[baseline_col]).mean()
                improvements.append((k, improvement))
        
        f.write("\nAverage Improvements Over Baseline:\n")
        for k, imp in improvements:
            f.write(f"Recall@{k}: {imp:.3f}\n")
    
    print(f"Summary: {summary_path}")

def run_single_experiment(
    llm_model: str,
    embedding_model: str,
    norm_type: str,
    is_pseudo: bool,
    num_samples: int,
    cache_dir: Path
) -> Dict[str, Any]:
    """Run a single experiment configuration and return results."""
    try:
        # Initialize corpus with current embedding model
        corpus = ProgrammingSolutionsCorpus(
            embedding_model_name=embedding_model
        )
        corpus.load(normalize_type=norm_type)
        
        # Initialize reranker
        reranker = HumanEvalPseudoRetrieval(
            corpus=corpus,
            llm_model_name=llm_model,
            cache_dir=str(cache_dir)
        )
        
        # Run evaluation
        results = reranker.evaluate_humaneval(
            k_values=K_VALUES,
            num_samples=num_samples,
            debug=True,
            is_pseudo=is_pseudo
        )
        
        # Record results
        result_entry = {
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "normalization_type": norm_type,
            "is_pseudo": is_pseudo,
        }
        
        # Add recall metrics
        for k, recall in results["baseline"].items():
            result_entry[f"baseline_recall@{k}"] = recall
        for k, recall in results["pseudo"].items():
            result_entry[f"pseudo_recall@{k}"] = recall
        
        return result_entry
        
    except Exception as e:
        print(f"Error running configuration: {e}")
        return {
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "normalization_type": norm_type,
            "is_pseudo": is_pseudo,
            "error": str(e)
        }

def run_experiments(output_dir: str = "pseudocode_experiment_results", num_samples: int = None):
    """Run all experiment configurations and save results."""
    output_dir = Path(output_dir)
    experiment_id = generate_experiment_id()
    print(f"Starting experiment run: {experiment_id}")
    
    # Generate all combinations of parameters
    configurations = list(product(
        LLM_MODELS,
        EMBEDDING_MODELS,
        NORMALIZATION_TYPES,
        IS_PSEUDO
    ))
    
    print(f"\nTotal configurations to test: {len(configurations)}")
    results = []
    
    for llm_model, embedding_model, norm_type, is_pseudo in tqdm(configurations):
        print(f"\n{'='*80}")
        print(f"Testing configuration:")
        print(f"LLM: {llm_model}")
        print(f"Embeddings: {embedding_model}")
        print(f"Normalization: {norm_type}")
        print(f"is_pseudo: {is_pseudo}")
        
        result = run_single_experiment(
            llm_model=llm_model,
            embedding_model=embedding_model,
            norm_type=norm_type,
            num_samples=num_samples,
            cache_dir=output_dir,
            is_pseudo = is_pseudo
        )
        
        results.append(result)
        # Save intermediate results
        save_results(results, experiment_id, output_dir)
    
    print("\nExperiments completed!")
    return experiment_id

def main():
    # Check for API key
    if "HF_API_KEY" not in os.environ:
        raise ValueError("Please set HF_API_KEY environment variable")
    
    # Run experiments
    experiment_id = run_experiments(
        output_dir="../results/humaneval_pseudocode",
        num_samples=None
    )
    
    print(f"\nExperiment {experiment_id} completed!")
    print("Check the experiment_results directory for detailed results and analysis.")

if __name__ == "__main__":
    main()
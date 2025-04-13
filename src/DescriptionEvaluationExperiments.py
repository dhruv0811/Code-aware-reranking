import os
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import argparse

from DescriptionBasedRetrieval import DescriptionBasedRetrieval

# Configuration constants
EXPERIMENT_NAME = "description_based_retrieval"

# Dataset options
DATASET_OPTIONS = [
    "openai_humaneval",
    "code-rag-bench/mbpp"
]

# Default dataset
DEFAULT_DATASET = "openai_humaneval"

# LLM models for generating descriptions
LLM_MODELS = [
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct"
]

# Embedding models for indexing and retrieval
EMBEDDING_MODELS = [
    "avsolatorio/GIST-large-Embedding-v0",
    "avsolatorio/GIST-Embedding-v0"
]

# Code normalization types
NORMALIZATION_TYPES = [
    "none",
    "docstring",
    "variables",
    "functions",
    "both"
]

# Evaluation metrics: Recall@K values
K_VALUES = [1, 5, 10, 25, 50, 100]

def generate_experiment_id() -> str:
    """Generate a unique identifier for the experiment run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{EXPERIMENT_NAME}_{timestamp}"

def save_results(results: List[Dict], experiment_id: str, output_dir: Path) -> None:
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
        f.write(f"=== {EXPERIMENT_NAME.replace('_', ' ').title()} Experiment Summary ===\n\n")
        f.write(f"Experiment ID: {experiment_id}\n")
        
        # Best configurations for each k value
        f.write("\nBest Configurations by Recall@K:\n")
        for k in K_VALUES:
            recall_col = f'recall@{k}'
            if recall_col in df.columns:
                df_filtered = df[~df[recall_col].isna()]
                if not df_filtered.empty:
                    best_idx = df_filtered[recall_col].idxmax()
                    best_config = df_filtered.loc[best_idx]
                    f.write(f"\nBest for Recall@{k} = {best_config[recall_col]:.4f}:\n")
                    f.write(f"- LLM: {best_config['llm_model']}\n")
                    f.write(f"- Embeddings: {best_config['embedding_model']}\n")
                    f.write(f"- Normalization: {best_config['normalization_type']}\n")
                    f.write(f"- Dataset: {best_config['dataset']}\n")
        
        # Best configurations for MRR
        f.write("\nBest Configurations by MRR:\n")
        if 'mrr' in df.columns:
            df_filtered = df[~df['mrr'].isna()]
            if not df_filtered.empty:
                best_idx = df_filtered['mrr'].idxmax()
                best_config = df_filtered.loc[best_idx]
                f.write(f"\nBest MRR = {best_config['mrr']:.4f}:\n")
                f.write(f"- LLM: {best_config['llm_model']}\n")
                f.write(f"- Embeddings: {best_config['embedding_model']}\n")
                f.write(f"- Normalization: {best_config['normalization_type']}\n")
                f.write(f"- Dataset: {best_config['dataset']}\n")
        
        # Average performance by model
        f.write("\nAverage Performance by LLM Model:\n")
        for llm in LLM_MODELS:
            model_df = df[df['llm_model'] == llm]
            if not model_df.empty:
                f.write(f"\n{llm.split('/')[-1]}:\n")
                for k in K_VALUES:
                    recall_col = f'recall@{k}'
                    if recall_col in model_df.columns:
                        avg_recall = model_df[recall_col].mean()
                        f.write(f"- Avg Recall@{k}: {avg_recall:.4f}\n")
                if 'mrr' in model_df.columns:
                    avg_mrr = model_df['mrr'].mean()
                    f.write(f"- Avg MRR: {avg_mrr:.4f}\n")

        # Average performance by normalization type
        f.write("\nAverage Performance by Normalization Type:\n")
        for norm_type in NORMALIZATION_TYPES:
            norm_df = df[df['normalization_type'] == norm_type]
            if not norm_df.empty:
                f.write(f"\n{norm_type}:\n")
                for k in K_VALUES:
                    recall_col = f'recall@{k}'
                    if recall_col in norm_df.columns:
                        avg_recall = norm_df[recall_col].mean()
                        f.write(f"- Avg Recall@{k}: {avg_recall:.4f}\n")
                if 'mrr' in norm_df.columns:
                    avg_mrr = norm_df['mrr'].mean()
                    f.write(f"- Avg MRR: {avg_mrr:.4f}\n")
    
    print(f"Summary: {summary_path}")

def run_single_experiment(
    llm_model: str,
    embedding_model: str,
    norm_type: str,
    dataset_name: str,
    num_samples: int,
    cache_dir: Path,
    force_rebuild: bool = False
) -> Dict[str, Any]:
    """Run a single experiment configuration with description-based retrieval and return results."""
    try:
        # Initialize the description-based retrieval system
        retriever = DescriptionBasedRetrieval(
            corpus_dataset="code-rag-bench/programming-solutions",
            embedding_model_name=embedding_model,
            llm_model_name=llm_model,
            normalize_type=norm_type,
            cache_dir=str(cache_dir / "retrieval_cache")
        )
        
        # Build the index (or load if it exists)
        retriever.build_index(force_rebuild=force_rebuild)
        
        # Run evaluation
        eval_results = retriever.evaluate(
            k_values=K_VALUES,
            num_samples=num_samples,
            dataset_name=dataset_name
        )
        
        # Record configuration and results
        result_entry = {
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "normalization_type": norm_type,
            "dataset": dataset_name,
            "mrr": eval_results["mrr"]
        }
        
        # Add recall metrics
        for k, recall in eval_results["recalls"].items():
            result_entry[f"recall@{k}"] = recall
            
        # Add NDCG metrics
        for k, ndcg in eval_results["ndcg"].items():
            result_entry[f"ndcg@{k}"] = ndcg
        
        # Save retrieved documents to a separate file
        if "retrieved_docs" in eval_results:
            docs_filename = f"desc_docs_{dataset_name.split('/')[-1]}_{llm_model.split('/')[-1]}_{embedding_model.split('/')[-1]}_{norm_type}.json"
            docs_path = cache_dir / "retrieved_docs" / docs_filename
            
            # Create directory if it doesn't exist
            (cache_dir / "retrieved_docs").mkdir(parents=True, exist_ok=True)
            
            with open(docs_path, 'w') as f:
                json.dump(eval_results["retrieved_docs"], f, indent=2)
                
            print(f"Saved retrieved documents to {docs_path}")
        
        return result_entry
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error running configuration: {e}")
        return {
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "normalization_type": norm_type,
            "dataset": dataset_name,
            "error": str(e)
        }

def run_experiments(
    output_dir: str = "results/description_retrieval",
    num_samples: int = None,
    datasets: List[str] = None,
    llm_models: List[str] = None,
    embedding_models: List[str] = None,
    norm_types: List[str] = None,
    force_rebuild: bool = False
):
    """Run experiments with different configurations and save results."""
    output_dir = Path(output_dir)
    experiment_id = generate_experiment_id()
    print(f"Starting experiment run: {experiment_id}")
    
    # Use default values if not specified
    datasets = datasets or [DEFAULT_DATASET]
    llm_models = llm_models or LLM_MODELS
    embedding_models = embedding_models or EMBEDDING_MODELS
    norm_types = norm_types or NORMALIZATION_TYPES
    
    # Generate all combinations of parameters
    configurations = list(product(
        llm_models,
        embedding_models,
        norm_types,
        datasets
    ))
    
    print(f"\nTotal configurations to test: {len(configurations)}")
    results = []
    
    for llm_model, embedding_model, norm_type, dataset in tqdm(configurations):
        print(f"\n{'='*80}")
        print(f"Testing configuration:")
        print(f"LLM: {llm_model}")
        print(f"Embeddings: {embedding_model}")
        print(f"Normalization: {norm_type}")
        print(f"Dataset: {dataset}")
        
        result = run_single_experiment(
            llm_model=llm_model,
            embedding_model=embedding_model,
            norm_type=norm_type,
            dataset_name=dataset,
            num_samples=num_samples,
            cache_dir=output_dir,
            force_rebuild=force_rebuild
        )
        
        results.append(result)
        # Save intermediate results
        save_results(results, experiment_id, output_dir)
    
    print("\nExperiments completed!")
    return experiment_id

def main():
    """Main function to run description-based retrieval experiments with command line options."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run description-based retrieval experiments")
    
    parser.add_argument("--output_dir", type=str, default="results/description_retrieval",
                        help="Directory to save experiment results")
    
    parser.add_argument("--dataset", type=str, choices=DATASET_OPTIONS, default=None,
                        help="Dataset to evaluate on")
    
    parser.add_argument("--llm", type=str, default=None,
                        help="LLM model to use for generating descriptions")
    
    parser.add_argument("--embedding", type=str, default=None,
                        help="Embedding model to use for indexing and retrieval")
    
    parser.add_argument("--norm_type", type=str, choices=NORMALIZATION_TYPES, default=None,
                        help="Code normalization type")
    
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (for testing)")
    
    parser.add_argument("--force_rebuild", action="store_true",
                        help="Force rebuilding the index even if it exists")
    
    args = parser.parse_args()
    
    # Check for API key
    if "HF_API_KEY" not in os.environ:
        raise ValueError("Please set HF_API_KEY environment variable")
    
    # Convert single arguments to lists for product()
    datasets = [args.dataset] if args.dataset else None
    llm_models = [args.llm] if args.llm else None
    embedding_models = [args.embedding] if args.embedding else None
    norm_types = [args.norm_type] if args.norm_type else None
    
    # Run experiments
    experiment_id = run_experiments(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        datasets=datasets,
        llm_models=llm_models,
        embedding_models=embedding_models,
        norm_types=norm_types,
        force_rebuild=args.force_rebuild
    )
    
    print(f"\nExperiment {experiment_id} completed!")
    print(f"Check {args.output_dir} for detailed results and analysis.")

if __name__ == "__main__":
    main()
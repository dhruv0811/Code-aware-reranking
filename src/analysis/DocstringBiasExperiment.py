import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np

# Import the existing corpus with normalization functionality
try:
    from src.Corpus import ProgrammingSolutionsCorpus
except ImportError:
    try:
        from Corpus import ProgrammingSolutionsCorpus
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from Corpus import ProgrammingSolutionsCorpus

class DocstringBiasComparison:
    """
    Compare different normalization techniques (none, docstring, functions)
    to analyze the impact on retrieval performance.
    """
    
    def __init__(
        self,
        embedding_model_names: List[str] = [
            "avsolatorio/GIST-large-Embedding-v0",
            "avsolatorio/GIST-Embedding-v0",
        ],
        output_dir: str = "results/docstring_comparison",
        dataset_name: str = "openai_humaneval",  # or "code-rag-bench/mbpp"
    ):
        """Initialize the docstring bias comparison experiment."""
        hf_api_key = os.getenv("HF_API_KEY")
        if hf_api_key is None:
            raise ValueError("Must provide HuggingFace API key as HF_API_KEY environment variable")
            
        self.embedding_model_names = embedding_model_names
        self.dataset_name = dataset_name
        self.normalization_types = ["none", "docstring", "functions"]
        
        # Determine the dataset parameters based on dataset name
        if dataset_name == "openai_humaneval":
            self.dataset_filter = "humaneval"
            self.query_key = "prompt"
            self.dataset_split = "test"
        elif dataset_name == "code-rag-bench/mbpp":
            self.dataset_filter = "mbpp"
            self.query_key = "text"
            self.dataset_split = "train"
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Initialize output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics we'll track
        self.stats = {}
    
    def _generate_experiment_id(self) -> str:
        """Generate a unique identifier for the experiment run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"normalization_comparison_{timestamp}"
    
    def run_comparison(
        self, 
        k_values: List[int] = [1, 5, 10, 25, 50, 100],
        num_samples: Optional[int] = None
    ):
        """
        Run the normalization comparison experiment.
        
        Args:
            k_values: List of k values for Recall@k metrics
            num_samples: Optional limit on number of queries to process
        """
        experiment_id = self._generate_experiment_id()
        print(f"Running experiment: {experiment_id}")
        print(f"Comparing normalization types: {', '.join(self.normalization_types)}")
        
        # Load dataset for queries
        dataset = load_dataset(self.dataset_name)
        
        queries = [item[self.query_key] for item in dataset[self.dataset_split]]
        task_ids = [item['task_id'] for item in dataset[self.dataset_split]]
        
        if num_samples:
            queries = queries[:num_samples]
            task_ids = task_ids[:num_samples]
        
        all_results = []
        
        # Run experiment for each embedding model
        for embedding_model_name in self.embedding_model_names:
            print(f"\n============================================")
            print(f"Testing embedding model: {embedding_model_name}")
            print(f"============================================")
            
            # Track results for each normalization type
            norm_results = {}
            
            for norm_type in self.normalization_types:
                print(f"\n--- Processing {norm_type} normalization ---")
                
                # Initialize corpus with this normalization
                corpus = ProgrammingSolutionsCorpus(
                    embedding_model_name=embedding_model_name
                )
                corpus.load(normalize_type=norm_type)
                
                # Track ranks for gold documents
                ranks = []
                
                # Track hit rates
                hits = {k: 0 for k in k_values}
                
                # Process each query
                print(f"Processing {len(queries)} queries...")
                for query_idx, (query, true_id) in enumerate(tqdm(zip(queries, task_ids), total=len(queries))):
                    true_id = str(true_id)
                    
                    # Get search results
                    search_results = corpus.search(query, k=max(k_values))
                    
                    # Find rank of the gold document (using 1-based indexing)
                    rank = None
                    max_k = max(k_values)
                    
                    for r, (doc, score) in enumerate(search_results):
                        doc_id = str(doc.metadata['index'])
                        if doc_id == true_id:
                            # Use 1-based indexing for ranks
                            rank = r + 1
                            break
                    
                    # If document wasn't found, assign rank as k+1
                    if rank is None:
                        rank = max_k + 1
                    
                    # Update hit statistics
                    for k in k_values:
                        if rank <= k:  # Changed to <= since ranks now start at 1
                            hits[k] += 1
                    
                    # Store rank
                    ranks.append(rank)
                
                # Calculate statistics
                avg_rank = np.mean(ranks) if ranks else float('nan')
                median_rank = np.median(ranks) if ranks else float('nan')
                
                # Calculate hit rates
                total_queries = len(queries)
                hit_rates = {k: hits/total_queries for k, hits in hits.items()}
                
                # Store results for this normalization type
                norm_results[norm_type] = {
                    "avg_rank": avg_rank,
                    "median_rank": median_rank,
                    "hit_rates": hit_rates
                }
                
                # Print summary for this normalization
                print(f"\nResults for {norm_type} normalization:")
                print(f"Average rank: {avg_rank:.2f}")
                print(f"Median rank: {median_rank:.2f}")
                
                # Print hit rates
                print("\nHit rates:")
                for k in k_values:
                    print(f"R@{k}: {hit_rates[k]:.3f}")
            
            # Compile model results
            model_results = {
                "embedding_model": embedding_model_name,
                "normalization_results": norm_results
            }
            
            all_results.append(model_results)
            
            # Print comparison table for this model
            self._print_comparison_table(embedding_model_name, norm_results, k_values)
        
        # Save all results
        self.stats = {
            "experiment_id": experiment_id,
            "dataset": self.dataset_name,
            "normalization_types": self.normalization_types,
            "model_results": all_results
        }
        
        self._save_results(experiment_id)
        
        return self.stats
    
    def _print_comparison_table(self, model_name, norm_results, k_values):
        """Print a comparison table for different normalization techniques."""
        print(f"\n{'='*100}")
        print(f"NORMALIZATION COMPARISON FOR {model_name}")
        print(f"{'='*100}\n")
        
        # Table header
        print(f"{'Normalization':<15} | {'Avg Rank':<15} | {'Median Rank':<15}")
        print("-" * 55)
        
        # Table rows
        for norm_type, results in norm_results.items():
            print(f"{norm_type:<15} | {results['avg_rank']:<15.2f} | {results['median_rank']:<15.2f}")
        
        # Hit rate tables
        for k in [1, 5, 10, 50]:
            if k not in k_values:
                continue
                
            print(f"\nR@{k} Hit Rates:")
            print(f"{'Normalization':<15} | {'Hit Rate':<15}")
            print("-" * 35)
            
            for norm_type, results in norm_results.items():
                hit_rate = results['hit_rates'][k]
                print(f"{norm_type:<15} | {hit_rate:<15.3f}")
    
    def _save_results(self, experiment_id: str):
        """Save experiment results to files."""
        results_path = self.output_dir / f"{experiment_id}_results.json"
        
        # Save to JSON file
        with open(results_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"Results saved to {results_path}")
        
        # Also save a CSV summary
        summary_path = self.output_dir / f"{experiment_id}_summary.csv"
        
        # Create DataFrame for CSV
        data = []
        
        for model_result in self.stats["model_results"]:
            model_name = model_result["embedding_model"]
            norm_results = model_result["normalization_results"]
            
            for norm_type, results in norm_results.items():
                # Basic metrics
                row = {
                    "embedding_model": model_name,
                    "normalization": norm_type,
                    "avg_rank": results["avg_rank"],
                    "median_rank": results["median_rank"],
                }
                
                # Add hit rates
                for k, rate in results["hit_rates"].items():
                    row[f"hit_rate_R@{k}"] = rate
                
                data.append(row)
        
        # Save as CSV
        df = pd.DataFrame(data)
        df.to_csv(summary_path, index=False)
        
        print(f"Summary saved to {summary_path}")


def run_normalization_comparison():
    """Run experiment to compare different normalization techniques."""
    # Check for API key
    if "HF_API_KEY" not in os.environ:
        raise ValueError("Please set HF_API_KEY environment variable")
    
    # Common parameters
    embedding_models = [
        "avsolatorio/GIST-large-Embedding-v0",
        "avsolatorio/GIST-Embedding-v0",
    ]
    dataset_name = "openai_humaneval"  # Change to "code-rag-bench/mbpp" for MBPP dataset
    output_dir = "./results/normalization_comparison"
    
    # For quick testing, use a small sample
    num_samples = None  # Set to a number like 10 for quick testing
    
    # Run experiment
    print(f"\n{'='*80}")
    print(f"=== Running normalization comparison experiment ===")
    print(f"{'='*80}\n")
    
    experiment = DocstringBiasComparison(
        embedding_model_names=embedding_models,
        output_dir=output_dir,
        dataset_name=dataset_name
    )
    experiment.run_comparison(num_samples=num_samples)
    
    print("\nExperiment completed! Check the output directory for detailed results.")


if __name__ == "__main__":
    run_normalization_comparison()
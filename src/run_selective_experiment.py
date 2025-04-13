import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from SelectiveNormalizationExperiment import SelectiveNormalizationExperiment

def main():
    parser = argparse.ArgumentParser(description="Run the selective normalization experiment")
    
    parser.add_argument("--embedding-model", type=str, 
                        default="avsolatorio/GIST-large-Embedding-v0",
                        help="Embedding model to use")
    
    parser.add_argument("--dataset", type=str, 
                        default="openai_humaneval", 
                        choices=["openai_humaneval", "code-rag-bench/mbpp"],
                        help="Dataset to use for queries")
    
    parser.add_argument("--k", type=int, default=100,
                        help="Maximum number of results to retrieve per query")
    
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Maximum number of queries to process (for debugging)")
    
    parser.add_argument("--output-dir", type=str, 
                        default="results/selective_normalization",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Check for HF API key
    if "HF_API_KEY" not in os.environ:
        raise ValueError("Please set HF_API_KEY environment variable")
    
    print(f"Running experiment with settings:")
    print(f"  Embedding Model: {args.embedding_model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  K: {args.k}")
    print(f"  Num Samples: {args.num_samples}")
    print(f"  Output Directory: {args.output_dir}")
    
    # Create experiment runner
    experiment = SelectiveNormalizationExperiment(
        embedding_model_name=args.embedding_model,
        output_dir=args.output_dir,
        dataset_name=args.dataset
    )
    
    # Run experiment
    print("\nRunning experiment...")
    results = experiment.run_experiment(
        k_values=[1, 5, 10, 25, 50, 100],
        num_samples=args.num_samples
    )
    
    # Print summary information
    print("\nExperiment Summary:")
    print(f"  Experiment ID: {results['experiment_id']}")
    print(f"  Dataset: {results['dataset']}")
    print(f"  Number of queries: {results['num_queries']}")
    
    print("\nAverage Ranks:")
    for norm_type, avg_rank in results['average_ranks'].items():
        print(f"  {norm_type}: {avg_rank:.2f}")
    
    print("\nMedian Ranks:")
    for norm_type, median_rank in results['median_ranks'].items():
        print(f"  {norm_type}: {median_rank:.2f}")
    
    print("\nRank Changes (Docstring Normalization):")
    docstring_changes = results['rank_changes']['docstring_normalized_gold']
    print(f"  Average Change: {docstring_changes['avg_change']:.2f}")
    print(f"  % Improved: {docstring_changes['improved_percent']:.2f}%")
    print(f"  % Worsened: {docstring_changes['worsened_percent']:.2f}%")
    
    print("\nRank Changes (Function Normalization):")
    function_changes = results['rank_changes']['function_normalized_gold']
    print(f"  Average Change: {function_changes['avg_change']:.2f}")
    print(f"  % Improved: {function_changes['improved_percent']:.2f}%")
    print(f"  % Worsened: {function_changes['worsened_percent']:.2f}%")
    
    print(f"\nDetailed results saved to:")
    print(f"  {args.output_dir}/{results['experiment_id']}_results.json")
    print(f"  {args.output_dir}/{results['experiment_id']}_summary.csv")
    print(f"  {args.output_dir}/{results['experiment_id']}_query_details.csv")

if __name__ == "__main__":
    main()
import argparse
import os
from pprint import pprint

from NormalizationExperiment import NormalizationComparisonExperiment

def main():
    parser = argparse.ArgumentParser(description="Run the normalization comparison experiment")
    
    parser.add_argument("--embedding-model", type=str, 
                        default="avsolatorio/GIST-large-Embedding-v0",
                        help="Embedding model to use")
    
    parser.add_argument("--dataset", type=str, 
                        default="openai_humaneval", 
                        choices=["openai_humaneval", "code-rag-bench/mbpp"],
                        help="Dataset to use for queries")
    
    parser.add_argument("--k", type=int, default=100,
                        help="Number of results to retrieve per query")
    
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Maximum number of documents to load (for debugging)")
    
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Maximum number of queries to process (for debugging)")
    
    parser.add_argument("--output-dir", type=str, 
                        default="results/normalization_experiment",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Check for HF API key
    if "HF_API_KEY" not in os.environ:
        raise ValueError("Please set HF_API_KEY environment variable")
    
    print(f"Running experiment with settings:")
    print(f"  Embedding Model: {args.embedding_model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  K: {args.k}")
    print(f"  Max Docs: {args.max_docs}")
    print(f"  Num Samples: {args.num_samples}")
    print(f"  Output Directory: {args.output_dir}")
    
    # Create experiment runner
    experiment = NormalizationComparisonExperiment(
        embedding_model_name=args.embedding_model,
        output_dir=args.output_dir,
        dataset_name=args.dataset
    )
    
    # Load corpus
    print("Loading corpus...")
    stats = experiment.load_corpus(max_docs=args.max_docs)
    print("Corpus statistics:")
    pprint(stats)
    
    # Run experiment
    print("Running experiment...")
    results = experiment.run_experiment(k=args.k, num_samples=args.num_samples)
    
    # Print results
    print("\nExperiment Results:")
    print("\nPreference for Original Documents (%):")
    for norm_type, percentage in results['original_preference'].items():
        print(f"  {norm_type}: {percentage:.2f}%")
    
    print("\nAverage Rank by Normalization Type:")
    for norm_type, avg_rank in results['average_ranks'].items():
        print(f"  {norm_type}: {avg_rank:.2f}")
    
    print("\nMedian Rank by Normalization Type:")
    for norm_type, median_rank in results['median_ranks'].items():
        print(f"  {norm_type}: {median_rank:.2f}")
    
    print("\nPreference for Irrelevant Original over Relevant Normalized (%):")
    for norm_type, percentage in results['irrelevant_preference'].items():
        print(f"  {norm_type}: {percentage:.2f}%")
    
    print(f"\nDetailed results saved to {args.output_dir}/{results['experiment_id']}_results.json")

if __name__ == "__main__":
    main()
import argparse
import os
from pprint import pprint

from NormalizationRerankingExperiment import NormalizationRerankingExperiment

def main():
    parser = argparse.ArgumentParser(description="Run the normalization comparison experiment with reranking")
    
    parser.add_argument("--embedding-model", type=str, 
                        default="avsolatorio/GIST-large-Embedding-v0",
                        help="Embedding model to use")
    
    parser.add_argument("--llm-model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="LLM model to use for reranking")
                        
    parser.add_argument("--dataset", type=str, 
                        default="openai_humaneval",
                        choices=["openai_humaneval", "code-rag-bench/mbpp"],
                        help="Dataset to use for queries")
    
    parser.add_argument("--k", type=int, default=100,
                        help="Number of results to retrieve per query")
                        
    parser.add_argument("--rerank-k", type=int, default=25,
                        help="Number of top results to rerank")
    
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Reranking weight (higher values give more weight to LLM descriptions)")
    
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Maximum number of documents to load (for debugging)")
    
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Maximum number of queries to process (for debugging)")
    
    parser.add_argument("--output-dir", type=str, 
                        default="results/normalization_reranking_experiment",
                        help="Directory to save results")
                        
    parser.add_argument("--cache-dir", type=str,
                        default="results/humaneval_best_saved",
                        help="Directory for cache files")
    
    args = parser.parse_args()
    
    # Check for HF API key
    if "HF_API_KEY" not in os.environ:
        raise ValueError("Please set HF_API_KEY environment variable")
    
    print(f"Running experiment with settings:")
    print(f"  Embedding Model: {args.embedding_model}")
    print(f"  LLM Model: {args.llm_model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  K: {args.k}")
    print(f"  Rerank K: {args.rerank_k}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Max Docs: {args.max_docs}")
    print(f"  Num Samples: {args.num_samples}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Cache Directory: {args.cache_dir}")
    
    # Create experiment runner
    experiment = NormalizationRerankingExperiment(
        embedding_model_name=args.embedding_model,
        llm_model_name=args.llm_model,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        cache_dir=args.cache_dir,
        alpha=args.alpha
    )
    
    # Load corpus
    print("Loading corpus...")
    stats = experiment.load_corpus(max_docs=args.max_docs)
    print("Corpus statistics:")
    pprint(stats)
    
    # Run experiment
    print("Running experiment...")
    results = experiment.run_experiment(
        k=args.k, 
        rerank_k=args.rerank_k,
        num_samples=args.num_samples
    )
    
    # Print results
    print("\nExperiment Results:")
    
    print("\nBase Ranking - Preference for Original Documents (%):")
    for norm_type, percentage in results['base_original_preference'].items():
        print(f"  {norm_type}: {percentage:.2f}%")
    
    print("\nReranked - Preference for Original Documents (%):")
    for norm_type, percentage in results['reranked_original_preference'].items():
        print(f"  {norm_type}: {percentage:.2f}%")
    
    print("\nBase Ranking - Average Rank by Normalization Type:")
    for norm_type, avg_rank in results['base_average_ranks'].items():
        print(f"  {norm_type}: {avg_rank:.2f}")
    
    print("\nReranked - Average Rank by Normalization Type:")
    for norm_type, avg_rank in results['reranked_average_ranks'].items():
        print(f"  {norm_type}: {avg_rank:.2f}")
    
    print("\nBase Ranking - Preference for Irrelevant Original over Relevant Normalized (%):")
    for norm_type, percentage in results['base_irrelevant_preference'].items():
        print(f"  {norm_type}: {percentage:.2f}%")
    
    print("\nReranked - Preference for Irrelevant Original over Relevant Normalized (%):")
    for norm_type, percentage in results['reranked_irrelevant_preference'].items():
        print(f"  {norm_type}: {percentage:.2f}%")
    
    print(f"\nDetailed results saved to {args.output_dir}/{results['experiment_id']}_results.json")

if __name__ == "__main__":
    main()
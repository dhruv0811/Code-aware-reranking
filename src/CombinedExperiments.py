import os
import json
from datetime import datetime
from itertools import product
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import load_dataset

from Corpus import ProgrammingSolutionsCorpus
from Pseudocode import HumanEvalPseudoRetrieval
from Reranking import ProgrammingSolutionsReranker, RetrievedCode

# Configuration constants
EXPERIMENT_NAME = "combined_pseudo_rerank"

# DATASET = "code-rag-bench/mbpp"
DATASET = "openai_humaneval"

LLM_MODELS = [
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct"
]

EMBEDDING_MODELS = [
    "avsolatorio/GIST-large-Embedding-v0",
    "avsolatorio/GIST-Embedding-v0"
]

NORMALIZATION_TYPES = [
    "none",
    "docstring",
    "variables",
    "functions",
    "both"
]

RERANK_K_VALUES = [25]
INITIAL_K = 100
ALPHA = 0.7

# Eval Recall@K Values
K_VALUES = [1, 5, 10, 25, 50, 100]

class CombinedRetriever:
    def __init__(
        self, 
        corpus,
        llm_model_name: str,
        cache_dir: str
    ):
        self.corpus = corpus
        self.pseudo_retriever = HumanEvalPseudoRetrieval(
            corpus=corpus,
            llm_model_name=llm_model_name,
            cache_dir=str(Path(cache_dir) / "pseudo")
        )
        self.reranker = ProgrammingSolutionsReranker(
            corpus=corpus,
            llm_model_name=llm_model_name,
            cache_dir=str(Path(cache_dir) / "rerank")
        )
    
    def retrieve_with_pseudocode(self, query: str, k: int, debug: bool = False):
        """Generate pseudocode for the query and retrieve documents."""
        pseudocode = self.pseudo_retriever.generate_pseudocode(query)
        
        if debug:
            print(f"\nOriginal query: {query[:200]}...")
            print(f"\nGenerated pseudocode: {pseudocode[:200]}...")
        
        results = self.corpus.search(pseudocode, k=k)
        
        retrieved_codes = [
            RetrievedCode(
                task_id=doc.metadata['index'],
                code=doc.page_content,
                initial_score=1.0 - score
            )
            for doc, score in results
        ]
        
        return retrieved_codes
    
    def retrieve_and_rerank(
        self,
        query: str,
        initial_k: int = 100,
        rerank_k: int = 25,
        alpha: float = 0.7,
        debug: bool = False
    ):
        """Combined pipeline: pseudocode retrieval then reranking."""
        # First retrieve using pseudocode
        if debug:
            print("Step 1: Retrieving documents using pseudocode...")
        
        pseudo_retrieved = self.retrieve_with_pseudocode(query, k=initial_k, debug=debug)
        
        if debug:
            print(f"Retrieved {len(pseudo_retrieved)} documents using pseudocode")
        
        # Sort by initial score and take top rerank_k items
        codes_to_rerank = sorted(
            pseudo_retrieved,
            key=lambda x: x.initial_score,
            reverse=True
        )[:rerank_k]
        
        if debug:
            print(f"\nStep 2: Reranking top {rerank_k} results using original query...")
        
        # Process each retrieved code for reranking
        for code_item in tqdm(codes_to_rerank, disable=not debug):
            if debug:
                print(f"\nProcessing code for task_id: {code_item.task_id}")
                print(f"Initial score: {code_item.initial_score:.4f}")
            
            # Generate code description
            code_item.code_description = self.reranker._generate_code_description(code_item.code)
            
            if debug:
                print(f"Generated description: {code_item.code_description}")
            
            # Calculate similarity with ORIGINAL query (not pseudocode)
            if code_item.code_description:
                description_similarity = self.corpus.compute_similarity(
                    query,  # Use original query
                    code_item.code_description
                )
            else:
                description_similarity = 0.0
            
            # Combine scores
            normalized_initial = code_item.initial_score * 100
            code_item.reranked_score = (1 - alpha) * normalized_initial + alpha * (description_similarity * 100)
            
            if debug:
                print(f"Description similarity with original query: {description_similarity:.4f}")
                print(f"Final reranked score: {code_item.reranked_score:.2f}")
        
        # Save cache after processing batch
        self.reranker.description_cache.save_cache(
            self.reranker.llm_model_name, 
            self.reranker.prompt_type
        )
        
        # Sort reranked results
        reranked_results = sorted(
            codes_to_rerank,
            key=lambda x: x.reranked_score,
            reverse=True
        )
        
        # Append remaining items sorted by initial score
        remaining = sorted(
            pseudo_retrieved[rerank_k:],
            key=lambda x: x.initial_score,
            reverse=True
        )
        reranked_results.extend(remaining)
        
        if debug:
            print("\nFinal combined retrieval results:")
            for i, result in enumerate(reranked_results[:5]):
                print(f"Rank {i+1}: task_id={result.task_id}, score={result.reranked_score:.2f}")
                if result.code_description:
                    print(f"Description: {result.code_description[:100]}...")
        
        return reranked_results
    
    def evaluate(
        self,
        k_values: List[int] = [1, 5, 10, 25, 50, 100],
        initial_k: int = 100,
        rerank_k: int = 25,
        alpha: float = 0.7,
        num_samples: int = None,
        debug: bool = False,
        dataset_name: str = "code-rag-bench/mbpp"
    ) -> Dict[str, Dict[int, float]]:
        """Evaluate the combined approach."""
        dataset = load_dataset(dataset_name)
        
        query_name = 'prompt' if dataset_name == "openai_humaneval" else 'text'
        key = 'train' if dataset_name == "code-rag-bench/mbpp" else 'test'
        
        queries = [item[query_name] for item in dataset[key]]
        task_ids = [item['task_id'] for item in dataset[key]]
        
        if num_samples:
            queries = queries[:num_samples]
            task_ids = task_ids[:num_samples]
        
        if debug:
            print(f"\nEvaluating {len(queries)} queries with combined approach")
            print("Sample task IDs to find:", task_ids[:3])
        
        # Track various retrieval methods for comparison
        baseline_recalls = {k: 0 for k in k_values}  # Standard embedding retrieval
        pseudo_recalls = {k: 0 for k in k_values}    # Pseudocode retrieval 
        combined_recalls = {k: 0 for k in k_values}  # Pseudocode + reranking
        
        max_k = max(k_values)
        
        for query_idx, (query, true_id) in enumerate(tqdm(zip(queries, task_ids), total=len(queries))):
            if debug:
                print(f"\n{'='*80}")
                print(f"Processing query {query_idx + 1}/{len(queries)}")
                print(f"True task ID: {true_id}")
            
            # 1. Get baseline results (original query)
            baseline_results = self.corpus.search(query, k=max_k)
            baseline_ids = [str(doc.metadata['index']) for doc, _ in baseline_results]
            
            # 2. Get pseudocode retrieval results 
            pseudocode = self.pseudo_retriever.generate_pseudocode(query)
            pseudo_results = self.corpus.search(pseudocode, k=max_k)
            pseudo_ids = [str(doc.metadata['index']) for doc, _ in pseudo_results]
            
            # 3. Get combined results (pseudocode retrieval + reranking)
            combined_results = self.retrieve_and_rerank(
                query=query, 
                initial_k=initial_k,
                rerank_k=rerank_k,
                alpha=alpha,
                debug=debug
            )
            combined_ids = [str(code.task_id) for code in combined_results]
            
            true_id = str(true_id)
            
            if debug:
                print("\nRESULTS COMPARISON:")
                print(f"Top 5 baseline IDs: {baseline_ids[:5]}")
                print(f"Top 5 pseudocode IDs: {pseudo_ids[:5]}")
                print(f"Top 5 combined IDs: {combined_ids[:5]}")
                print(f"Looking for true_id: {true_id}")
            
            # Update recall counts
            for k in k_values:
                if true_id in baseline_ids[:k]:
                    baseline_recalls[k] += 1
                if true_id in pseudo_ids[:k]:
                    pseudo_recalls[k] += 1
                if true_id in combined_ids[:k]:
                    combined_recalls[k] += 1
        
        # Calculate final recall metrics
        num_queries = len(queries)
        baseline_recalls = {k: count/num_queries for k, count in baseline_recalls.items()}
        pseudo_recalls = {k: count/num_queries for k, count in pseudo_recalls.items()}
        combined_recalls = {k: count/num_queries for k, count in combined_recalls.items()}
        
        return {
            "baseline": baseline_recalls,
            "pseudocode": pseudo_recalls,
            "combined": combined_recalls
        }

def generate_experiment_id() -> str:
    """Generate a unique identifier for the experiment run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{EXPERIMENT_NAME}_{timestamp}"

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
        f.write("=== Combined Approach Experiment Summary ===\n\n")
        f.write(f"Dataset: {DATASET}\n")
        f.write(f"Experiment ID: {experiment_id}\n\n")
        
        # Best configurations for each k value
        f.write("Best Configurations by Recall@K:\n")
        for k in K_VALUES:
            combined_col = f'combined_recall@{k}'
            df_filtered = df[~df[combined_col].isna()]
            if not df_filtered.empty:
                best_idx = df_filtered[combined_col].idxmax()
                best_config = df_filtered.loc[best_idx]
                f.write(f"\nBest for Recall@{k} = {best_config[combined_col]:.3f}:\n")
                f.write(f"- LLM: {best_config['llm_model']}\n")
                f.write(f"- Embeddings: {best_config['embedding_model']}\n")
                f.write(f"- Normalization: {best_config['normalization_type']}\n")
                f.write(f"- Rerank K: {best_config['rerank_k']}\n")
                f.write(f"- Alpha: {best_config['alpha']}\n")
        
        # Average improvements
        f.write("\nAverage Improvements:\n")
        f.write("K\tBaseline\tPseudocode\tCombined\tImprovement over Baseline\tImprovement over Pseudocode\n")
        for k in K_VALUES:
            baseline_col = f'baseline_recall@{k}'
            pseudo_col = f'pseudocode_recall@{k}'
            combined_col = f'combined_recall@{k}'
            
            if all(col in df.columns for col in [baseline_col, pseudo_col, combined_col]):
                avg_baseline = df[baseline_col].mean()
                avg_pseudo = df[pseudo_col].mean()
                avg_combined = df[combined_col].mean()
                
                imp_over_baseline = avg_combined - avg_baseline
                imp_over_pseudo = avg_combined - avg_pseudo
                
                f.write(f"{k}\t{avg_baseline:.3f}\t{avg_pseudo:.3f}\t{avg_combined:.3f}\t{imp_over_baseline:.3f}\t{imp_over_pseudo:.3f}\n")
    
    print(f"Summary: {summary_path}")

def run_single_experiment(
    llm_model: str,
    embedding_model: str,
    norm_type: str,
    rerank_k: int,
    num_samples: int,
    cache_dir: Path
) -> Dict[str, Any]:
    """Run a single experiment with the combined approach."""
    try:
        # Initialize corpus with current embedding model
        corpus = ProgrammingSolutionsCorpus(
            embedding_model_name=embedding_model
        )
        corpus.load(normalize_type=norm_type)
        
        # Initialize combined retriever
        combined_retriever = CombinedRetriever(
            corpus=corpus,
            llm_model_name=llm_model,
            cache_dir=str(cache_dir)
        )
        
        # Run evaluation
        results = combined_retriever.evaluate(
            k_values=K_VALUES,
            initial_k=INITIAL_K,
            rerank_k=rerank_k,
            alpha=ALPHA,
            num_samples=num_samples,
            debug=False,
            dataset_name=DATASET
        )
        
        # Record results
        result_entry = {
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "normalization_type": norm_type,
            "rerank_k": rerank_k,
            "initial_k": INITIAL_K,
            "alpha": ALPHA
        }
        
        # Add recall metrics
        for method in ["baseline", "pseudocode", "combined"]:
            for k, recall in results[method].items():
                result_entry[f'{method}_recall@{k}'] = recall
        
        return result_entry
        
    except Exception as e:
        print(f"Error running configuration: {e}")
        import traceback
        traceback.print_exc()
        return {
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "normalization_type": norm_type,
            "rerank_k": rerank_k,
            "initial_k": INITIAL_K,
            "alpha": ALPHA,
            "error": str(e)
        }

def run_experiments(output_dir: str = "combined_experiment_results", num_samples: int = None):
    """Run all combined experiment configurations and save results."""
    output_dir = Path(output_dir)
    experiment_id = generate_experiment_id()
    print(f"Starting combined experiment run: {experiment_id}")
    
    # Generate all combinations of parameters
    configurations = list(product(
        LLM_MODELS,
        EMBEDDING_MODELS,
        NORMALIZATION_TYPES,
        RERANK_K_VALUES
    ))
    
    print(f"\nTotal configurations to test: {len(configurations)}")
    results = []
    
    for llm_model, embedding_model, norm_type, rerank_k in tqdm(configurations):
        print(f"\n{'='*80}")
        print(f"Testing configuration:")
        print(f"LLM: {llm_model}")
        print(f"Embeddings: {embedding_model}")
        print(f"Normalization: {norm_type}")
        print(f"Rerank K: {rerank_k}")
        
        result = run_single_experiment(
            llm_model=llm_model,
            embedding_model=embedding_model,
            norm_type=norm_type,
            rerank_k=rerank_k,
            num_samples=num_samples,
            cache_dir=output_dir
        )
        
        results.append(result)
        # Save intermediate results
        save_results(results, experiment_id, output_dir)
    
    print("\nCombined experiments completed!")
    return experiment_id

def main():
    # Check for API key
    if "HF_API_KEY" not in os.environ:
        raise ValueError("Please set HF_API_KEY environment variable")
    
    # Run experiments
    experiment_id = run_experiments(
        output_dir="./results/humaneval_combined_pseudo_rerank",
        num_samples=None
    )
    
    print(f"\nExperiment {experiment_id} completed!")
    print("Check the results directory for detailed results and analysis.")

if __name__ == "__main__":
    main()
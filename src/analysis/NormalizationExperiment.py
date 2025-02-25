import os
import json
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

from NormalizationCorpus import NormalizationCorpus

class NormalizationComparisonExperiment:
    """Experiment runner to analyze how normalization affects retrieval preferences."""
    
    def __init__(
        self, 
        embedding_model_name: str = "avsolatorio/GIST-large-Embedding-v0",
        output_dir: str = "results/normalization_experiment",
        dataset_name: str = "openai_humaneval"  # or "code-rag-bench/mbpp"
    ):
        """Initialize the experiment runner."""
        hf_api_key = os.getenv("HF_API_KEY")
        if hf_api_key is None:
            raise ValueError("Must provide HuggingFace API key")
            
        self.embedding_model_name = embedding_model_name
        self.dataset_name = dataset_name
        
        # Determine the dataset filter based on dataset name
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
        
        # Initialize corpus
        self.corpus = NormalizationCorpus(embedding_model_name=embedding_model_name)
        
        # Statistics we'll track
        self.original_preference_stats = {}
        self.norm_rank_stats = {}
        self.irrelevant_preference_stats = {}
    
    def _generate_experiment_id(self) -> str:
        """Generate a unique identifier for the experiment run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"norm_experiment_{timestamp}"
    
    def load_corpus(self, max_docs: Optional[int] = None):
        """Load the corpus with different normalization levels."""
        return self.corpus.load(dataset_filter=self.dataset_filter, max_docs=max_docs)
    
    def run_experiment(self, k: int = 100, num_samples: Optional[int] = None):
        """Run the normalization comparison experiment.
        
        Args:
            k: Number of results to retrieve per query
            num_samples: Optional limit on number of queries to process
        """
        experiment_id = self._generate_experiment_id()
        print(f"Running experiment: {experiment_id}")
        
        # Load dataset for queries
        dataset = load_dataset(self.dataset_name)
        
        queries = [item[self.query_key] for item in dataset[self.dataset_split]]
        task_ids = [item['task_id'] for item in dataset[self.dataset_split]]
        
        if num_samples:
            queries = queries[:num_samples]
            task_ids = task_ids[:num_samples]
        
        # Initialize statistics trackers
        normalization_types = ["none", "docstring", "functions", "variables", "both"]
        
        # Prefer original stats: % of times original is ranked better than normalized
        prefer_original_stats = {norm_type: 0 for norm_type in normalization_types if norm_type != "none"}
        
        # Average rank stats by normalization type
        rank_stats = {norm_type: [] for norm_type in normalization_types}
        
        # Irrelevant preference stats: % of times irrelevant original is ranked better than relevant normalized
        irrelevant_preferred_stats = {norm_type: 0 for norm_type in normalization_types if norm_type != "none"}
        total_comparisons = {norm_type: 0 for norm_type in normalization_types if norm_type != "none"}
        
        # Store detailed results for each query
        query_results = []
        
        # Process each query
        print(f"Processing {len(queries)} queries...")
        for query_idx, (query, true_id) in enumerate(tqdm(zip(queries, task_ids), total=len(queries))):
            true_id = str(true_id)
            
            # Get search results
            search_results = self.corpus.search(query, k=k)
            
            # Group results by task_id and normalization type
            results_by_task_and_norm = defaultdict(dict)
            ranks_by_norm = defaultdict(list)
            
            for rank, (doc, score) in enumerate(search_results):
                task_id = doc.metadata['index']
                norm_type = doc.metadata['norm_type']
                
                # Track the rank of each normalization type
                ranks_by_norm[norm_type].append(rank)
                
                # Group by task_id and normalization_type
                results_by_task_and_norm[task_id][norm_type] = rank
            
            # Track average rank for each normalization type
            for norm_type in normalization_types:
                if ranks_by_norm[norm_type]:
                    rank_stats[norm_type].extend(ranks_by_norm[norm_type])
            
            # Compare original vs. normalized for the same task_id
            for task_id, norm_ranks in results_by_task_and_norm.items():
                if "none" in norm_ranks:
                    original_rank = norm_ranks["none"]
                    
                    # Check each normalization type
                    for norm_type in ["docstring", "functions", "variables", "both"]:
                        if norm_type in norm_ranks:
                            normalized_rank = norm_ranks[norm_type]
                            
                            # If original rank is better (smaller), increment preference counter
                            if original_rank < normalized_rank:
                                prefer_original_stats[norm_type] += 1
            
            # Compare irrelevant original vs. relevant normalized
            for irrelevant_task_id, irrelevant_ranks in results_by_task_and_norm.items():
                # Skip the true task_id (it's relevant by definition)
                if irrelevant_task_id == true_id:
                    continue
                    
                # Only consider if the irrelevant task has an original version
                if "none" not in irrelevant_ranks:
                    continue
                    
                irrelevant_original_rank = irrelevant_ranks["none"]
                
                # Check if the true task_id has normalized versions
                if true_id in results_by_task_and_norm:
                    for norm_type in ["docstring", "functions", "variables", "both"]:
                        if norm_type in results_by_task_and_norm[true_id]:
                            relevant_normalized_rank = results_by_task_and_norm[true_id][norm_type]
                            
                            # If irrelevant original ranks better than relevant normalized
                            if irrelevant_original_rank < relevant_normalized_rank:
                                irrelevant_preferred_stats[norm_type] += 1
                            
                            total_comparisons[norm_type] += 1
            
            # Store detailed results for this query
            query_result = {
                "query_id": query_idx,
                "query": query,
                "true_id": true_id,
                "ranks_by_norm_type": {k: v for k, v in ranks_by_norm.items()},
                "results_by_task_and_norm": {str(k): v for k, v in results_by_task_and_norm.items()}
            }
            query_results.append(query_result)
        
        # Calculate final statistics
        total_queries = len(queries)
        
        # Convert raw counts to percentages
        for norm_type in ["docstring", "functions", "variables", "both"]:
            # Calculate percentage preference for original
            prefer_original_stats[norm_type] = (prefer_original_stats[norm_type] / total_queries) * 100
            
            # Calculate percentage preference for irrelevant original over relevant normalized
            if total_comparisons[norm_type] > 0:
                irrelevant_preferred_stats[norm_type] = (irrelevant_preferred_stats[norm_type] / total_comparisons[norm_type]) * 100
            else:
                irrelevant_preferred_stats[norm_type] = 0.0
        
        # Calculate average ranks
        avg_ranks = {norm_type: np.mean(ranks) if ranks else float('nan') for norm_type, ranks in rank_stats.items()}
        median_ranks = {norm_type: np.median(ranks) if ranks else float('nan') for norm_type, ranks in rank_stats.items()}
        
        # Save experiment results
        self.original_preference_stats = prefer_original_stats
        self.norm_rank_stats = {
            'average': avg_ranks,
            'median': median_ranks,
            'counts': {norm_type: len(ranks) for norm_type, ranks in rank_stats.items()}
        }
        self.irrelevant_preference_stats = irrelevant_preferred_stats
        
        # Save results to file
        self._save_results(experiment_id, query_results)
        
        return {
            'experiment_id': experiment_id,
            'original_preference': prefer_original_stats,
            'average_ranks': avg_ranks,
            'median_ranks': median_ranks,
            'irrelevant_preference': irrelevant_preferred_stats,
            'total_queries': total_queries
        }

    def _save_results(self, experiment_id: str, query_results: List[Dict]):
        """Save experiment results to files."""
        results_path = self.output_dir / f"{experiment_id}_results.json"
        
        # Compile results
        summary = {
            'experiment_id': experiment_id,
            'embedding_model': self.embedding_model_name,
            'dataset': self.dataset_name,
            'stats': {
                'original_preference': self.original_preference_stats,
                'norm_rank_stats': self.norm_rank_stats,
                'irrelevant_preference': self.irrelevant_preference_stats
            }
        }
        
        # Create detailed results
        detailed_results = {
            'summary': summary,
            'queries': query_results
        }
        
        # Save to JSON file
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"Results saved to {results_path}")
        
        # Also save a CSV summary
        summary_path = self.output_dir / f"{experiment_id}_summary.csv"
        
        # Create DataFrame for CSV
        data = []
        
        # Original preference stats
        for norm_type, percentage in self.original_preference_stats.items():
            data.append({
                'metric': 'original_preference_pct',
                'normalization': norm_type,
                'value': percentage
            })
        
        # Average rank stats
        for norm_type, avg_rank in self.norm_rank_stats['average'].items():
            data.append({
                'metric': 'average_rank',
                'normalization': norm_type,
                'value': avg_rank
            })
        
        # Median rank stats
        for norm_type, median_rank in self.norm_rank_stats['median'].items():
            data.append({
                'metric': 'median_rank',
                'normalization': norm_type,
                'value': median_rank
            })
        
        # Irrelevant preference stats
        for norm_type, percentage in self.irrelevant_preference_stats.items():
            data.append({
                'metric': 'irrelevant_preference_pct',
                'normalization': norm_type,
                'value': percentage
            })
        
        # Save as CSV
        df = pd.DataFrame(data)
        df.to_csv(summary_path, index=False)
        
        print(f"Summary saved to {summary_path}")
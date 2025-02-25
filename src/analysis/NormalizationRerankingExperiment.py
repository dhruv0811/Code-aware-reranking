import os
import json
import math
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from huggingface_hub import InferenceClient
from time import sleep

from NormalizationCorpus import NormalizationCorpus

DESCRIPTION_PROMPT = """Analyze this code and provide a concise description of what it does.
Focus on the main functionality, algorithm, and purpose.
Keep the description under 100 words.

Code:
{}

Provide only the description with no additional text or formatting."""

class DescriptionCache:
    def __init__(self, cache_dir: str = "cache/reranker"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.modified = False
    
    def _get_cache_key(self, code: str, model_name: str, prompt_type: str) -> str:
        """Generate a unique key for the code-model-prompt combination."""
        # Use hash to handle potentially long code strings
        code_hash = hashlib.md5(code.encode()).hexdigest()
        return f"{code_hash}_{model_name}_{prompt_type}"
    
    def _get_cache_file(self, model_name: str, prompt_type: str) -> Path:
        """Get the cache file path for a specific model and prompt type."""
        safe_model_name = model_name.replace('/', '_')
        return self.cache_dir / f"cache_{safe_model_name}_{prompt_type}.pkl"
    
    def load_cache(self, model_name: str, prompt_type: str):
        """Load the cache file for a specific model and prompt type."""
        cache_file = self._get_cache_file(model_name, prompt_type)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"Loaded cache from {cache_file} with {len(self.cache)} entries")
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.cache = {}
        else:
            self.cache = {}
    
    def save_cache(self, model_name: str, prompt_type: str):
        """Save the cache to disk if it has been modified."""
        if self.modified:
            cache_file = self._get_cache_file(model_name, prompt_type)
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
                print(f"Saved cache to {cache_file} with {len(self.cache)} entries")
                self.modified = False
            except Exception as e:
                print(f"Error saving cache: {e}")
    
    def get(self, code: str, model_name: str, prompt_type: str) -> str:
        """Retrieve a cached description."""
        key = self._get_cache_key(code, model_name, prompt_type)
        return self.cache.get(key)
    
    def set(self, code: str, model_name: str, prompt_type: str, description: str):
        """Store a description in the cache."""
        key = self._get_cache_key(code, model_name, prompt_type)
        self.cache[key] = description
        self.modified = True

class NormalizationRerankingExperiment:
    """Experiment to compare normalization metrics between base ranking and reranking."""
    
    def __init__(
        self, 
        embedding_model_name: str = "avsolatorio/GIST-large-Embedding-v0",
        llm_model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
        output_dir: str = "results/normalization_reranking_experiment",
        dataset_name: str = "openai_humaneval",  # or "code-rag-bench/mbpp"
        cache_dir: str = "cache/reranker",
        alpha: float = 0.7  # Reranking weight
    ):
        """Initialize the experiment."""
        hf_api_key = os.getenv("HF_API_KEY")
        if hf_api_key is None:
            raise ValueError("Must provide HuggingFace API key")
            
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.dataset_name = dataset_name
        self.alpha = alpha
        
        # Initialize LLM client
        self.client = InferenceClient(api_key=hf_api_key)
        
        # Initialize description cache
        self.cache_dir = Path(cache_dir)
        self.description_cache = DescriptionCache(cache_dir)
        self.prompt_type = "basic"
        
        # Load cache for the current model and prompt type
        self.description_cache.load_cache(self.llm_model_name, self.prompt_type)
        
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
        self.base_original_preference_stats = {}
        self.base_norm_rank_stats = {}
        self.base_irrelevant_preference_stats = {}
        
        self.reranked_original_preference_stats = {}
        self.reranked_norm_rank_stats = {}
        self.reranked_irrelevant_preference_stats = {}
    
    def _generate_experiment_id(self) -> str:
        """Generate a unique identifier for the experiment run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"norm_reranking_experiment_{timestamp}"
    
    def load_corpus(self, max_docs: Optional[int] = None):
        """Load the corpus with different normalization levels."""
        return self.corpus.load(dataset_filter=self.dataset_filter, max_docs=max_docs)
    
    def _generate_code_description(self, code: str, max_retries: int = 3) -> str:
        """Generate a description of the code using the LLM."""
        # Check cache first
        cached_description = self.description_cache.get(
            code, 
            self.llm_model_name, 
            self.prompt_type
        )
        
        if cached_description is not None:
            return cached_description
            
        prompt = DESCRIPTION_PROMPT.format(code)
        
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=100,
                    temperature=0.3
                )
                
                description = completion.choices[0].message.content.strip()
                
                # Cache the generated description
                self.description_cache.set(
                    code,
                    self.llm_model_name,
                    self.prompt_type,
                    description
                )
                
                return description
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to get code description: {e}")
                    return ""
                wait_time = 2 ** attempt
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...")
                sleep(wait_time)
        
        return ""
    
    def _rerank_results(self, query: str, search_results, rerank_k: int = 10):
        """Rerank the top-k results using code descriptions."""
        # Extract the documents, scores, and their contents
        docs_with_scores = []
        
        # Process only the top-k for reranking
        for i, (doc, score) in enumerate(search_results[:rerank_k]):
            # Convert score to similarity (1.0 - distance)
            initial_score = 1.0 - score
            
            docs_with_scores.append({
                'doc': doc,
                'initial_score': initial_score,
                'code': doc.page_content,
                'code_description': None,
                'reranked_score': None
            })
        
        # Generate descriptions and compute reranked scores
        print(f"Reranking top {rerank_k} results using code descriptions...")
        for item in tqdm(docs_with_scores, desc="Generating descriptions"):
            item['code_description'] = self._generate_code_description(item['code'])
            
            if item['code_description']:
                description_similarity = self.corpus.compute_similarity(query, item['code_description'])
            else:
                description_similarity = 0.0
            
            # Apply reranking formula
            normalized_initial = item['initial_score'] * 100
            item['reranked_score'] = (1 - self.alpha) * normalized_initial + self.alpha * (description_similarity * 100)
        
        # Save cache after processing batch
        self.description_cache.save_cache(self.llm_model_name, self.prompt_type)
        
        # Sort by reranked score
        reranked_docs = sorted(docs_with_scores, key=lambda x: x['reranked_score'], reverse=True)
        
        # Add remaining documents without reranking
        remaining_docs = []
        for i, (doc, score) in enumerate(search_results[rerank_k:]):
            initial_score = 1.0 - score
            remaining_docs.append({
                'doc': doc,
                'initial_score': initial_score,
                'code': doc.page_content,
                'code_description': None,
                'reranked_score': initial_score * 100  # Just for consistency
            })
        
        # Combine reranked and remaining results
        reranked_docs.extend(remaining_docs)
        
        return reranked_docs
    
    def run_experiment(self, k: int = 100, rerank_k: int = 10, num_samples: Optional[int] = None):
        """Run the normalization comparison experiment with reranking.
        
        Args:
            k: Number of results to retrieve per query
            rerank_k: Number of top results to rerank
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
        
        # Base ranking stats
        base_prefer_original_stats = {norm_type: 0 for norm_type in normalization_types if norm_type != "none"}
        base_rank_stats = {norm_type: [] for norm_type in normalization_types}
        base_irrelevant_preferred_stats = {norm_type: 0 for norm_type in normalization_types if norm_type != "none"}
        base_total_comparisons = {norm_type: 0 for norm_type in normalization_types if norm_type != "none"}
        
        # Reranked stats
        reranked_prefer_original_stats = {norm_type: 0 for norm_type in normalization_types if norm_type != "none"}
        reranked_rank_stats = {norm_type: [] for norm_type in normalization_types}
        reranked_irrelevant_preferred_stats = {norm_type: 0 for norm_type in normalization_types if norm_type != "none"}
        reranked_total_comparisons = {norm_type: 0 for norm_type in normalization_types if norm_type != "none"}
        
        # Store detailed results for each query
        query_results = []
        
        # Process each query
        print(f"Processing {len(queries)} queries...")
        for query_idx, (query, true_id) in enumerate(tqdm(zip(queries, task_ids), total=len(queries))):
            true_id = str(true_id)
            
            # Get search results
            search_results = self.corpus.search(query, k=k)
            
            # Apply reranking
            reranked_docs = self._rerank_results(query, search_results, rerank_k=rerank_k)
            
            # Extract base ranks and reranked ranks
            base_ranks_by_doc = {}
            for rank, (doc, score) in enumerate(search_results):
                doc_id = f"{doc.metadata['index']}_{doc.metadata['norm_type']}"
                base_ranks_by_doc[doc_id] = rank
            
            reranked_ranks_by_doc = {}
            for rank, doc_info in enumerate(reranked_docs):
                doc = doc_info['doc']
                doc_id = f"{doc.metadata['index']}_{doc.metadata['norm_type']}"
                reranked_ranks_by_doc[doc_id] = rank
            
            # Group results by task_id and normalization type for base ranking
            base_results_by_task_and_norm = defaultdict(dict)
            base_ranks_by_norm = defaultdict(list)
            
            for doc, score in search_results:
                task_id = doc.metadata['index']
                norm_type = doc.metadata['norm_type']
                doc_id = f"{task_id}_{norm_type}"
                rank = base_ranks_by_doc[doc_id]
                
                # Track the rank of each normalization type
                base_ranks_by_norm[norm_type].append(rank)
                
                # Group by task_id and normalization_type
                base_results_by_task_and_norm[task_id][norm_type] = rank
            
            # Group results by task_id and normalization type for reranked
            reranked_results_by_task_and_norm = defaultdict(dict)
            reranked_ranks_by_norm = defaultdict(list)
            
            for doc_info in reranked_docs:
                doc = doc_info['doc']
                task_id = doc.metadata['index']
                norm_type = doc.metadata['norm_type']
                doc_id = f"{task_id}_{norm_type}"
                rank = reranked_ranks_by_doc[doc_id]
                
                # Track the rank of each normalization type
                reranked_ranks_by_norm[norm_type].append(rank)
                
                # Group by task_id and normalization_type
                reranked_results_by_task_and_norm[task_id][norm_type] = rank
            
            # Track average rank for each normalization type
            for norm_type in normalization_types:
                if base_ranks_by_norm[norm_type]:
                    base_rank_stats[norm_type].extend(base_ranks_by_norm[norm_type])
                if reranked_ranks_by_norm[norm_type]:
                    reranked_rank_stats[norm_type].extend(reranked_ranks_by_norm[norm_type])
            
            # Compare original vs. normalized for the same task_id - BASE RANKING
            for task_id, norm_ranks in base_results_by_task_and_norm.items():
                if "none" in norm_ranks:
                    original_rank = norm_ranks["none"]
                    
                    # Check each normalization type
                    for norm_type in ["docstring", "functions", "variables", "both"]:
                        if norm_type in norm_ranks:
                            normalized_rank = norm_ranks[norm_type]
                            
                            # If original rank is better (smaller), increment preference counter
                            if original_rank < normalized_rank:
                                base_prefer_original_stats[norm_type] += 1
            
            # Compare original vs. normalized for the same task_id - RERANKED
            for task_id, norm_ranks in reranked_results_by_task_and_norm.items():
                if "none" in norm_ranks:
                    original_rank = norm_ranks["none"]
                    
                    # Check each normalization type
                    for norm_type in ["docstring", "functions", "variables", "both"]:
                        if norm_type in norm_ranks:
                            normalized_rank = norm_ranks[norm_type]
                            
                            # If original rank is better (smaller), increment preference counter
                            if original_rank < normalized_rank:
                                reranked_prefer_original_stats[norm_type] += 1
            
            # Compare irrelevant original vs. relevant normalized - BASE RANKING
            for irrelevant_task_id, irrelevant_ranks in base_results_by_task_and_norm.items():
                # Skip the true task_id (it's relevant by definition)
                if irrelevant_task_id == true_id:
                    continue
                    
                # Only consider if the irrelevant task has an original version
                if "none" not in irrelevant_ranks:
                    continue
                    
                irrelevant_original_rank = irrelevant_ranks["none"]
                
                # Check if the true task_id has normalized versions
                if true_id in base_results_by_task_and_norm:
                    for norm_type in ["docstring", "functions", "variables", "both"]:
                        if norm_type in base_results_by_task_and_norm[true_id]:
                            relevant_normalized_rank = base_results_by_task_and_norm[true_id][norm_type]
                            
                            # If irrelevant original ranks better than relevant normalized
                            if irrelevant_original_rank < relevant_normalized_rank:
                                base_irrelevant_preferred_stats[norm_type] += 1
                            
                            base_total_comparisons[norm_type] += 1
            
            # Compare irrelevant original vs. relevant normalized - RERANKED
            for irrelevant_task_id, irrelevant_ranks in reranked_results_by_task_and_norm.items():
                # Skip the true task_id (it's relevant by definition)
                if irrelevant_task_id == true_id:
                    continue
                    
                # Only consider if the irrelevant task has an original version
                if "none" not in irrelevant_ranks:
                    continue
                    
                irrelevant_original_rank = irrelevant_ranks["none"]
                
                # Check if the true task_id has normalized versions
                if true_id in reranked_results_by_task_and_norm:
                    for norm_type in ["docstring", "functions", "variables", "both"]:
                        if norm_type in reranked_results_by_task_and_norm[true_id]:
                            relevant_normalized_rank = reranked_results_by_task_and_norm[true_id][norm_type]
                            
                            # If irrelevant original ranks better than relevant normalized
                            if irrelevant_original_rank < relevant_normalized_rank:
                                reranked_irrelevant_preferred_stats[norm_type] += 1
                            
                            reranked_total_comparisons[norm_type] += 1
            
            # Store detailed results for this query
            query_result = {
                "query_id": query_idx,
                "query": query,
                "true_id": true_id,
                "base_ranks_by_norm_type": {k: v for k, v in base_ranks_by_norm.items()},
                "base_results_by_task_and_norm": {str(k): v for k, v in base_results_by_task_and_norm.items()},
                "reranked_ranks_by_norm_type": {k: v for k, v in reranked_ranks_by_norm.items()},
                "reranked_results_by_task_and_norm": {str(k): v for k, v in reranked_results_by_task_and_norm.items()}
            }
            query_results.append(query_result)
        
        # Calculate final statistics
        total_queries = len(queries)
        
        # Convert raw counts to percentages for base ranking
        for norm_type in ["docstring", "functions", "variables", "both"]:
            # Calculate percentage preference for original
            base_prefer_original_stats[norm_type] = (base_prefer_original_stats[norm_type] / total_queries) * 100
            
            # Calculate percentage preference for irrelevant original over relevant normalized
            if base_total_comparisons[norm_type] > 0:
                base_irrelevant_preferred_stats[norm_type] = (base_irrelevant_preferred_stats[norm_type] / base_total_comparisons[norm_type]) * 100
            else:
                base_irrelevant_preferred_stats[norm_type] = 0.0
        
        # Convert raw counts to percentages for reranking
        for norm_type in ["docstring", "functions", "variables", "both"]:
            # Calculate percentage preference for original
            reranked_prefer_original_stats[norm_type] = (reranked_prefer_original_stats[norm_type] / total_queries) * 100
            
            # Calculate percentage preference for irrelevant original over relevant normalized
            if reranked_total_comparisons[norm_type] > 0:
                reranked_irrelevant_preferred_stats[norm_type] = (reranked_irrelevant_preferred_stats[norm_type] / reranked_total_comparisons[norm_type]) * 100
            else:
                reranked_irrelevant_preferred_stats[norm_type] = 0.0
        
        # Calculate average ranks for base
        base_avg_ranks = {norm_type: np.mean(ranks) if ranks else float('nan') for norm_type, ranks in base_rank_stats.items()}
        base_median_ranks = {norm_type: np.median(ranks) if ranks else float('nan') for norm_type, ranks in base_rank_stats.items()}
        
        # Calculate average ranks for reranked
        reranked_avg_ranks = {norm_type: np.mean(ranks) if ranks else float('nan') for norm_type, ranks in reranked_rank_stats.items()}
        reranked_median_ranks = {norm_type: np.median(ranks) if ranks else float('nan') for norm_type, ranks in reranked_rank_stats.items()}
        
        # Save experiment results
        self.base_original_preference_stats = base_prefer_original_stats
        self.base_norm_rank_stats = {
            'average': base_avg_ranks,
            'median': base_median_ranks,
            'counts': {norm_type: len(ranks) for norm_type, ranks in base_rank_stats.items()}
        }
        self.base_irrelevant_preference_stats = base_irrelevant_preferred_stats
        
        self.reranked_original_preference_stats = reranked_prefer_original_stats
        self.reranked_norm_rank_stats = {
            'average': reranked_avg_ranks,
            'median': reranked_median_ranks,
            'counts': {norm_type: len(ranks) for norm_type, ranks in reranked_rank_stats.items()}
        }
        self.reranked_irrelevant_preference_stats = reranked_irrelevant_preferred_stats
        
        # Save results to file
        self._save_results(experiment_id, query_results)
        
        return {
            'experiment_id': experiment_id,
            'base_original_preference': base_prefer_original_stats,
            'base_average_ranks': base_avg_ranks,
            'base_median_ranks': base_median_ranks,
            'base_irrelevant_preference': base_irrelevant_preferred_stats,
            'reranked_original_preference': reranked_prefer_original_stats,
            'reranked_average_ranks': reranked_avg_ranks,
            'reranked_median_ranks': reranked_median_ranks,
            'reranked_irrelevant_preference': reranked_irrelevant_preferred_stats,
            'total_queries': total_queries
        }

    def _save_results(self, experiment_id: str, query_results: List[Dict]):
        """Save experiment results to files."""
        results_path = self.output_dir / f"{experiment_id}_results.json"
        
        # Compile results
        summary = {
            'experiment_id': experiment_id,
            'embedding_model': self.embedding_model_name,
            'llm_model': self.llm_model_name,
            'dataset': self.dataset_name,
            'alpha': self.alpha,
            'stats': {
                'base_original_preference': self.base_original_preference_stats,
                'base_norm_rank_stats': self.base_norm_rank_stats,
                'base_irrelevant_preference': self.base_irrelevant_preference_stats,
                'reranked_original_preference': self.reranked_original_preference_stats,
                'reranked_norm_rank_stats': self.reranked_norm_rank_stats,
                'reranked_irrelevant_preference': self.reranked_irrelevant_preference_stats
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
        
        # Original preference stats - base
        for norm_type, percentage in self.base_original_preference_stats.items():
            data.append({
                'metric': 'original_preference_pct',
                'normalization': norm_type,
                'ranking': 'base',
                'value': percentage
            })
        
        # Original preference stats - reranked
        for norm_type, percentage in self.reranked_original_preference_stats.items():
            data.append({
                'metric': 'original_preference_pct',
                'normalization': norm_type,
                'ranking': 'reranked',
                'value': percentage
            })
        
        # Average rank stats - base
        for norm_type, avg_rank in self.base_norm_rank_stats['average'].items():
            data.append({
                'metric': 'average_rank',
                'normalization': norm_type,
                'ranking': 'base',
                'value': avg_rank
            })
        
        # Average rank stats - reranked
        for norm_type, avg_rank in self.reranked_norm_rank_stats['average'].items():
            data.append({
                'metric': 'average_rank',
                'normalization': norm_type,
                'ranking': 'reranked',
                'value': avg_rank
            })
        
        # Median rank stats - base
        for norm_type, median_rank in self.base_norm_rank_stats['median'].items():
            data.append({
                'metric': 'median_rank',
                'normalization': norm_type,
                'ranking': 'base',
                'value': median_rank
            })
        
        # Median rank stats - reranked
        for norm_type, median_rank in self.reranked_norm_rank_stats['median'].items():
            data.append({
                'metric': 'median_rank',
                'normalization': norm_type,
                'ranking': 'reranked',
                'value': median_rank
            })
        
        # Irrelevant preference stats - base
        for norm_type, percentage in self.base_irrelevant_preference_stats.items():
            data.append({
                'metric': 'irrelevant_preference_pct',
                'normalization': norm_type,
                'ranking': 'base',
                'value': percentage
            })
        
        # Irrelevant preference stats - reranked
        for norm_type, percentage in self.reranked_irrelevant_preference_stats.items():
            data.append({
                'metric': 'irrelevant_preference_pct',
                'normalization': norm_type,
                'ranking': 'reranked',
                'value': percentage
            })
        
        # Save as CSV
        df = pd.DataFrame(data)
        df.to_csv(summary_path, index=False)
        
        print(f"Summary saved to {summary_path}")
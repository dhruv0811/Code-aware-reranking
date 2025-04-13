import os
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

# Import the existing corpus with normalization functionality
try:
    from src.Corpus import ProgrammingSolutionsCorpus, CodeNormalizer
except ImportError:
    try:
        from Corpus import ProgrammingSolutionsCorpus, CodeNormalizer
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from Corpus import ProgrammingSolutionsCorpus, CodeNormalizer

class SelectiveNormalizationExperiment:
    """
    Experiment to analyze the impact of selective normalization of gold documents.
    
    This experiment compares how normalizing only the gold standard document
    affects its ranking when competing with non-normalized documents in the corpus.
    
    Three scenarios are tested:
    1. Baseline: No normalization in corpus
    2. Docstring-normalized gold: Only gold document has docstring removed
    3. Function-normalized gold: Only gold document has function names normalized
    """
    
    def __init__(
        self,
        embedding_model_name: str = "avsolatorio/GIST-large-Embedding-v0",
        output_dir: str = "results/selective_normalization",
        dataset_name: str = "openai_humaneval",  # or "code-rag-bench/mbpp"
    ):
        """Initialize the selective normalization experiment."""
        hf_api_key = os.getenv("HF_API_KEY")
        if hf_api_key is None:
            raise ValueError("Must provide HuggingFace API key as HF_API_KEY environment variable")
            
        self.embedding_model_name = embedding_model_name
        self.dataset_name = dataset_name
        
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
        
        # Initialize corpus and dataset
        self.corpus = None
        self.dataset = None
        self.queries = None
        self.task_ids = None
        
        # Reference to code normalizer for application to specific documents
        self.code_normalizer = CodeNormalizer()
        
        # Statistics we'll track
        self.stats = {}
    
    def _generate_experiment_id(self) -> str:
        """Generate a unique identifier for the experiment run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"selective_normalization_{timestamp}"
    
    def load_data(self, num_samples: Optional[int] = None):
        """
        Load the dataset and initialize the corpus with no normalization.
        
        Args:
            num_samples: Optional limit on number of queries to process
        """
        print(f"Loading dataset: {self.dataset_name}")
        self.dataset = load_dataset(self.dataset_name)
        
        # Extract queries and task IDs
        self.queries = [item[self.query_key] for item in self.dataset[self.dataset_split]]
        self.task_ids = [item['task_id'] for item in self.dataset[self.dataset_split]]
        
        if num_samples:
            self.queries = self.queries[:num_samples]
            self.task_ids = self.task_ids[:num_samples]
        
        print(f"Loaded {len(self.queries)} queries")
        
        # Initialize the base corpus with no normalization
        print(f"Initializing corpus with embedding model: {self.embedding_model_name}")
        self.corpus = ProgrammingSolutionsCorpus(
            embedding_model_name=self.embedding_model_name
        )
        self.corpus.load(normalize_type="none")
    
    def create_modified_corpus_for_query(self, query_idx: int, norm_type: str):
        """
        Create a modified corpus where only the gold document for the given query is normalized.
        
        Args:
            query_idx: Index of the query in self.queries
            norm_type: Type of normalization to apply to gold document ("docstring" or "functions")
            
        Returns:
            Modified corpus with selectively normalized gold document
        """
        if self.corpus is None:
            raise ValueError("Base corpus not initialized. Call load_data() first.")
        
        # Get task ID for this query
        task_id = self.task_ids[query_idx]
        
        # First, search for the gold document using the corpus's search functionality
        print(f"Finding gold document for task ID {task_id}...")
        
        # We'll try to find the gold document by searching for it
        max_results = 1000  # A large number to ensure we find it
        search_results = self.corpus.search(
            f"task_id:{task_id}",  # This is just a trick to get documents - the actual query doesn't matter
            k=max_results
        )
        
        # Find the gold document among search results
        gold_doc = None
        gold_doc_text = None
        gold_doc_metadata = None
        
        for doc, _ in search_results:
            if str(doc.metadata.get('index')) == str(task_id):
                gold_doc = doc
                gold_doc_text = doc.page_content
                gold_doc_metadata = doc.metadata
                break
        
        if gold_doc is None:
            # Try a different approach - using the FAISS docstore directly
            print(f"Gold document not found in initial search, trying direct FAISS access...")
            try:
                # Try to access the docstore directly if available
                if hasattr(self.corpus.vectorstore, 'docstore'):
                    # Iterate through all documents in the docstore
                    for doc_id, doc in self.corpus.vectorstore.docstore._dict.items():
                        if str(doc.metadata.get('index')) == str(task_id):
                            gold_doc = doc
                            gold_doc_text = doc.page_content
                            gold_doc_metadata = doc.metadata
                            break
            except Exception as e:
                print(f"Error accessing docstore: {e}")
        
        if gold_doc is None:
            # Try getting all docs with a large k value
            try:
                print(f"Trying to find gold document with large k search...")
                all_results = self.corpus.search("", k=10000)  # Try to get all documents
                for doc, _ in all_results:
                    if str(doc.metadata.get('index')) == str(task_id):
                        gold_doc = doc
                        gold_doc_text = doc.page_content
                        gold_doc_metadata = doc.metadata
                        break
            except Exception as e:
                print(f"Error during large k search: {e}")
        
        if gold_doc is None:
            raise ValueError(f"Gold document for task ID {task_id} not found in corpus.")
        
        print(f"Found gold document for task ID {task_id}")
        
        # Determine task type for normalization
        task_type = "humaneval" if self.dataset_filter == "humaneval" else "mbpp"
        
        # Normalize the gold document
        normalized_gold_doc = self.corpus.normalize_code(gold_doc_text, normalize_type=norm_type, task=task_type)
        
        # Now we need to get all documents to recreate the corpus
        print(f"Retrieving all documents from corpus...")
        all_docs = []
        
        try:
            # Get all documents with a large k value
            all_docs = self.corpus.search("", k=10000)  # Try to get all documents
        except Exception as e:
            print(f"Error retrieving all documents: {e}")
            # Try an alternative approach
            try:
                if hasattr(self.corpus.vectorstore, 'docstore'):
                    for doc_id, doc in self.corpus.vectorstore.docstore._dict.items():
                        all_docs.append((doc, 0.0))  # Add dummy score
            except Exception as e2:
                print(f"Error with alternative document retrieval: {e2}")
        
        print(f"Retrieved {len(all_docs)} documents from corpus")
        
        # Now prepare documents and metadata for the new corpus
        documents = []
        metadatas = []
        
        for doc, _ in all_docs:
            if str(doc.metadata.get('index')) == str(task_id):
                # This is the gold document, use the normalized version
                documents.append(normalized_gold_doc)
            else:
                documents.append(doc.page_content)
            
            metadatas.append(doc.metadata)
        
        # Create a new FAISS index with these documents
        from langchain_community.vectorstores import FAISS
        
        print(f"Creating new vector store with {len(documents)} documents...")
        
        # Check if documents and metadatas have the same length
        if len(documents) != len(metadatas):
            print(f"Warning: Document count ({len(documents)}) doesn't match metadata count ({len(metadatas)})")
            # Ensure they have the same length
            min_len = min(len(documents), len(metadatas))
            documents = documents[:min_len]
            metadatas = metadatas[:min_len]
        
        modified_vectorstore = FAISS.from_texts(
            documents,
            self.corpus.embedding_model,
            metadatas=metadatas
        )
        
        # Create a modified corpus with this vector store
        modified_corpus = ProgrammingSolutionsCorpus(
            embedding_model_name=self.embedding_model_name
        )
        modified_corpus.vectorstore = modified_vectorstore
        
        return modified_corpus
    
    def run_experiment(
        self,
        k_values: List[int] = [1, 5, 10, 25, 50, 100],
        num_samples: Optional[int] = None
    ):
        """
        Run the selective normalization experiment.
        
        For each query:
        1. Search the baseline corpus (no normalization)
        2. Create a corpus where only the gold document has docstring normalization
        3. Create a corpus where only the gold document has function normalization
        4. Compare the ranks of the gold document in each scenario
        
        Args:
            k_values: List of k values for Recall@k metrics
            num_samples: Optional limit on number of queries to process
        """
        experiment_id = self._generate_experiment_id()
        print(f"Running experiment: {experiment_id}")
        
        # Load data if not already loaded
        if self.corpus is None:
            self.load_data(num_samples=num_samples)
        
        # Metrics we'll track
        base_ranks = []  # Ranks in baseline corpus (no normalization)
        docstring_ranks = []  # Ranks when only gold has docstring normalization
        function_ranks = []  # Ranks when only gold has function normalization
        
        # Track hits at various k values
        base_hits = {k: 0 for k in k_values}
        docstring_hits = {k: 0 for k in k_values}
        function_hits = {k: 0 for k in k_values}
        
        # Process each query
        print(f"Processing {len(self.queries)} queries...")
        for query_idx, (query, true_id) in enumerate(tqdm(zip(self.queries, self.task_ids), total=len(self.queries))):
            true_id = str(true_id)
            
            # Get rank in baseline corpus
            base_rank = self._get_gold_document_rank(query, true_id, self.corpus, max_k=max(k_values))
            base_ranks.append(base_rank)
            
            # Update hit statistics for baseline
            for k in k_values:
                if base_rank <= k:
                    base_hits[k] += 1
            
            # Create corpus with only gold document having docstring normalization
            docstring_corpus = self.create_modified_corpus_for_query(query_idx, "docstring")
            docstring_rank = self._get_gold_document_rank(query, true_id, docstring_corpus, max_k=max(k_values))
            docstring_ranks.append(docstring_rank)
            
            # Update hit statistics for docstring normalization
            for k in k_values:
                if docstring_rank <= k:
                    docstring_hits[k] += 1
            
            # Create corpus with only gold document having function normalization
            function_corpus = self.create_modified_corpus_for_query(query_idx, "functions")
            function_rank = self._get_gold_document_rank(query, true_id, function_corpus, max_k=max(k_values))
            function_ranks.append(function_rank)
            
            # Update hit statistics for function normalization
            for k in k_values:
                if function_rank <= k:
                    function_hits[k] += 1
        
        # Calculate statistics
        total_queries = len(self.queries)
        
        # Average ranks
        avg_base_rank = np.mean(base_ranks) if base_ranks else float('nan')
        avg_docstring_rank = np.mean(docstring_ranks) if docstring_ranks else float('nan')
        avg_function_rank = np.mean(function_ranks) if function_ranks else float('nan')
        
        # Median ranks
        median_base_rank = np.median(base_ranks) if base_ranks else float('nan')
        median_docstring_rank = np.median(docstring_ranks) if docstring_ranks else float('nan')
        median_function_rank = np.median(function_ranks) if function_ranks else float('nan')
        
        # Calculate hit rates
        base_hit_rates = {k: hits/total_queries for k, hits in base_hits.items()}
        docstring_hit_rates = {k: hits/total_queries for k, hits in docstring_hits.items()}
        function_hit_rates = {k: hits/total_queries for k, hits in function_hits.items()}
        
        # Calculate rank changes
        docstring_rank_changes = [base - doc for base, doc in zip(base_ranks, docstring_ranks)]
        function_rank_changes = [base - func for base, func in zip(base_ranks, function_ranks)]
        
        # Percentage of queries where rank improved/worsened
        docstring_improved = sum(1 for change in docstring_rank_changes if change < 0) / total_queries * 100
        docstring_worsened = sum(1 for change in docstring_rank_changes if change > 0) / total_queries * 100
        docstring_unchanged = sum(1 for change in docstring_rank_changes if change == 0) / total_queries * 100
        
        function_improved = sum(1 for change in function_rank_changes if change < 0) / total_queries * 100
        function_worsened = sum(1 for change in function_rank_changes if change > 0) / total_queries * 100
        function_unchanged = sum(1 for change in function_rank_changes if change == 0) / total_queries * 100
        
        # Compile results
        self.stats = {
            "experiment_id": experiment_id,
            "embedding_model": self.embedding_model_name,
            "dataset": self.dataset_name,
            "num_queries": total_queries,
            "average_ranks": {
                "baseline": avg_base_rank,
                "docstring_normalized_gold": avg_docstring_rank,
                "function_normalized_gold": avg_function_rank
            },
            "median_ranks": {
                "baseline": median_base_rank,
                "docstring_normalized_gold": median_docstring_rank,
                "function_normalized_gold": median_function_rank
            },
            "hit_rates": {
                "baseline": base_hit_rates,
                "docstring_normalized_gold": docstring_hit_rates,
                "function_normalized_gold": function_hit_rates
            },
            "rank_changes": {
                "docstring_normalized_gold": {
                    "avg_change": np.mean(docstring_rank_changes),
                    "worsened_percent": docstring_improved,
                    "improved_percent": docstring_worsened,
                    "unchanged_percent": docstring_unchanged
                },
                "function_normalized_gold": {
                    "avg_change": np.mean(function_rank_changes),
                    "improved_percent": function_improved,
                    "worsened_percent": function_worsened,
                    "unchanged_percent": function_unchanged
                }
            },
            "per_query_details": [
                {
                    "query_idx": i,
                    "task_id": task_id,
                    "base_rank": base_ranks[i],
                    "docstring_rank": docstring_ranks[i],
                    "function_rank": function_ranks[i],
                    "docstring_change": docstring_rank_changes[i],
                    "function_change": function_rank_changes[i]
                }
                for i, task_id in enumerate(self.task_ids)
            ]
        }
        
        # Print results summary
        self._print_results_summary()
        
        # Save results
        self._save_results(experiment_id)
        
        return self.stats
    
    def _get_gold_document_rank(self, query: str, true_id: str, corpus, max_k: int = 100) -> int:
        """
        Get the rank of the gold document for a query.
        
        Args:
            query: The query string
            true_id: The ID of the gold document
            corpus: The corpus to search
            max_k: Maximum number of results to retrieve
            
        Returns:
            Rank of the gold document (1-based indexing)
        """
        try:
            # Get search results
            search_results = corpus.search(query, k=max_k)
            
            # Find rank of the gold document (using 1-based indexing)
            for r, (doc, score) in enumerate(search_results):
                # Handle the case where index might be in different formats
                doc_id = str(doc.metadata.get('index', ''))
                if doc_id == true_id:
                    # Use 1-based indexing for ranks
                    return r + 1
            
            # If document wasn't found, assign rank as max_k + 1
            return max_k + 1
        except Exception as e:
            print(f"Error finding rank for document {true_id}: {e}")
            # If there's an error, use the worst possible rank
            return max_k + 1
    
    def _print_results_summary(self):
        """Print a summary of the experiment results."""
        print(f"\n{'='*100}")
        print(f"SELECTIVE NORMALIZATION EXPERIMENT RESULTS")
        print(f"{'='*100}\n")
        
        # Print average ranks
        print("Average Ranks (lower is better):")
        avg_ranks = self.stats["average_ranks"]
        print(f"  Baseline: {avg_ranks['baseline']:.2f}")
        print(f"  Docstring Normalized Gold: {avg_ranks['docstring_normalized_gold']:.2f}")
        print(f"  Function Normalized Gold: {avg_ranks['function_normalized_gold']:.2f}")
        
        # Print median ranks
        print("\nMedian Ranks:")
        median_ranks = self.stats["median_ranks"]
        print(f"  Baseline: {median_ranks['baseline']:.2f}")
        print(f"  Docstring Normalized Gold: {median_ranks['docstring_normalized_gold']:.2f}")
        print(f"  Function Normalized Gold: {median_ranks['function_normalized_gold']:.2f}")
        
        # Print rank changes
        print("\nRank Changes from Baseline:")
        
        docstring_changes = self.stats["rank_changes"]["docstring_normalized_gold"]
        print("  Docstring Normalization:")
        print(f"    Average Change: {docstring_changes['avg_change']:.2f}")
        print(f"    Improved: {docstring_changes['improved_percent']:.2f}%")
        print(f"    Worsened: {docstring_changes['worsened_percent']:.2f}%")
        print(f"    Unchanged: {docstring_changes['unchanged_percent']:.2f}%")
        
        function_changes = self.stats["rank_changes"]["function_normalized_gold"]
        print("  Function Normalization:")
        print(f"    Average Change: {function_changes['avg_change']:.2f}")
        print(f"    Improved: {function_changes['improved_percent']:.2f}%")
        print(f"    Worsened: {function_changes['worsened_percent']:.2f}%")
        print(f"    Unchanged: {function_changes['unchanged_percent']:.2f}%")
        
        # Print hit rates for a few k values
        for k in [1, 5, 10, 50]:
            if str(k) not in self.stats["hit_rates"]["baseline"]:
                continue
                
            print(f"\nR@{k} Hit Rates:")
            print(f"  Baseline: {self.stats['hit_rates']['baseline'][str(k)]:.3f}")
            print(f"  Docstring Normalized Gold: {self.stats['hit_rates']['docstring_normalized_gold'][str(k)]:.3f}")
            print(f"  Function Normalized Gold: {self.stats['hit_rates']['function_normalized_gold'][str(k)]:.3f}")
    
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
        
        baseline_data = {
            "normalization_type": "baseline",
            "avg_rank": self.stats["average_ranks"]["baseline"],
            "median_rank": self.stats["median_ranks"]["baseline"],
        }
        
        for k, rate in self.stats["hit_rates"]["baseline"].items():
            baseline_data[f"hit_rate_R@{k}"] = rate
            
        data.append(baseline_data)
        
        # Add docstring data
        docstring_data = {
            "normalization_type": "docstring_normalized_gold",
            "avg_rank": self.stats["average_ranks"]["docstring_normalized_gold"],
            "median_rank": self.stats["median_ranks"]["docstring_normalized_gold"],
            "avg_change": self.stats["rank_changes"]["docstring_normalized_gold"]["avg_change"],
            "improved_percent": self.stats["rank_changes"]["docstring_normalized_gold"]["improved_percent"],
            "worsened_percent": self.stats["rank_changes"]["docstring_normalized_gold"]["worsened_percent"],
            "unchanged_percent": self.stats["rank_changes"]["docstring_normalized_gold"]["unchanged_percent"],
        }
        
        for k, rate in self.stats["hit_rates"]["docstring_normalized_gold"].items():
            docstring_data[f"hit_rate_R@{k}"] = rate
            
        data.append(docstring_data)
        
        # Add function data
        function_data = {
            "normalization_type": "function_normalized_gold",
            "avg_rank": self.stats["average_ranks"]["function_normalized_gold"],
            "median_rank": self.stats["median_ranks"]["function_normalized_gold"],
            "avg_change": self.stats["rank_changes"]["function_normalized_gold"]["avg_change"],
            "improved_percent": self.stats["rank_changes"]["function_normalized_gold"]["improved_percent"],
            "worsened_percent": self.stats["rank_changes"]["function_normalized_gold"]["worsened_percent"],
            "unchanged_percent": self.stats["rank_changes"]["function_normalized_gold"]["unchanged_percent"],
        }
        
        for k, rate in self.stats["hit_rates"]["function_normalized_gold"].items():
            function_data[f"hit_rate_R@{k}"] = rate
            
        data.append(function_data)
        
        # Save as CSV
        df = pd.DataFrame(data)
        df.to_csv(summary_path, index=False)
        
        print(f"Summary saved to {summary_path}")
        
        # Also save per-query details
        details_path = self.output_dir / f"{experiment_id}_query_details.csv"
        details_df = pd.DataFrame(self.stats["per_query_details"])
        details_df.to_csv(details_path, index=False)
        
        print(f"Query details saved to {details_path}")


if __name__ == "__main__":
    # Run the experiment with default settings
    experiment = SelectiveNormalizationExperiment()
    experiment.run_experiment(num_samples=10)  # Change num_samples as needed for testing
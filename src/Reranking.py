import math
import os 
import pickle
import hashlib
from tqdm import tqdm
from time import sleep
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from datasets import load_dataset
from huggingface_hub import InferenceClient

DESCRIPTION_PROMPT = """Analyze this code and provide a concise description of what it does.
Focus on the main functionality, algorithm, and purpose.
Keep the description under 100 words.

Code:
{}

Provide only the description with no additional text or formatting."""

@dataclass
class RetrievedCode:
    task_id: str
    code: str
    initial_score: float
    code_description: str = None
    reranked_score: float = None

class DescriptionCache:
    def __init__(self, cache_dir: str = "pseudocode_cache/reranker"):
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

class ProgrammingSolutionsReranker:
    def __init__(
        self,
        corpus,
        llm_model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
        cache_dir: str = "description_cache"
    ):
        hf_api_key = os.getenv("HF_API_KEY")
        if hf_api_key is None:
            raise ValueError("Must provide HuggingFace API key")
            
        self.corpus = corpus
        self.llm_model_name = llm_model_name
        self.client = InferenceClient(api_key=hf_api_key)
        self.description_cache = DescriptionCache(cache_dir)
        self.prompt_type = "basic"  # Default prompt type
        
        # Load the cache for the current model and prompt type
        self.description_cache.load_cache(self.llm_model_name, self.prompt_type)

    def _generate_code_description(self, code: str, max_retries: int = 3) -> str:
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

    def retrieve_and_rerank(
        self,
        query: str,
        initial_k: int = 100,
        rerank_k: int = 5,
        alpha: float = 0.7,
        debug: bool = True
    ) -> List[RetrievedCode]:
        if debug:
            print(f"\nProcessing query: {query[:200]}...")
        
        results = self.corpus.search(query, k=initial_k)
        
        if debug:
            print(f"Retrieved {len(results)} initial results")
            print("Sample retrieved scores:", [round(score, 4) for _, score in results[:5]])
        
        retrieved_codes = [
            RetrievedCode(
                task_id=doc.metadata['index'],
                code=doc.page_content,
                initial_score=1.0 - score
            )
            for doc, score in results
        ]
        
        codes_to_rerank = sorted(
            retrieved_codes,
            key=lambda x: x.initial_score,
            reverse=True
        )[:rerank_k]
        
        print(f"\nReranking top {rerank_k} results using code descriptions...")
        for code_item in tqdm(codes_to_rerank):
            if debug:
                print(f"\nProcessing code for task_id: {code_item.task_id}")
                print(f"Initial score: {code_item.initial_score:.4f}")
            
            code_item.code_description = self._generate_code_description(code_item.code)
            
            if debug:
                print("Code: ", {code_item.code[:100]})
            if debug:
                print(f"Generated description: {code_item.code_description}")
            
            if code_item.code_description:
                description_similarity = self.corpus.compute_similarity(
                    query, 
                    code_item.code_description
                )
            else:
                description_similarity = 0.0
            
            normalized_initial = code_item.initial_score * 100
            code_item.reranked_score = (1 - alpha) * normalized_initial + alpha * (description_similarity * 100)
            
            if debug:
                print(f"Description similarity: {description_similarity:.4f}")
                print(f"Final reranked score: {code_item.reranked_score:.2f}")
        
        # Save cache after processing batch
        self.description_cache.save_cache(self.llm_model_name, self.prompt_type)
        
        reranked_results = sorted(
            codes_to_rerank,
            key=lambda x: x.reranked_score,
            reverse=True
        )
        
        remaining = sorted(
            retrieved_codes[rerank_k:],
            key=lambda x: x.initial_score,
            reverse=True
        )
        reranked_results.extend(remaining)
        
        # if debug:
        #     print("\nFinal reranking results:")
        #     for i, result in enumerate(reranked_results[:5]):
        #         print(f"Rank {i+1}: task_id={result.task_id}, score={result.reranked_score:.2f}")
        #         if result.code_description:
        #             print(f"Description: {result.code_description}")
        
        return reranked_results
    
    def _compute_ndcg_at_k(self, retrieved_ids: List[str], true_id: str, k: int) -> float:
        """
        Compute NDCG@k for a single query.
        For our case, relevance is binary (1 if it matches true_id, 0 otherwise).
        """
        # Get relevance scores for top k results
        relevance = [1 if doc_id == true_id else 0 for doc_id in retrieved_ids[:k]]
        
        # Calculate DCG@k
        dcg = 0
        for i, rel in enumerate(relevance):
            # Using log base 2 (i+2 because ranks start at 1, and log(1) = 0)
            dcg += rel / math.log2(i + 2)
        
        # Calculate IDCG@k (ideal DCG)
        # For binary relevance, ideal ranking would have the relevant doc at position 1
        idcg = 1.0  # 1/log2(1+1) = 1
        
        # Return NDCG@k
        return dcg / idcg if idcg > 0 else 0.0
    
    def _compute_mrr(self, retrieved_ids: List[str], true_id: str) -> float:
        """
        Compute Mean Reciprocal Rank for a single query.
        Returns 1/rank of the first relevant document, or 0 if not found.
        """
        try:
            # Find the first occurrence of true_id
            rank = retrieved_ids.index(true_id) + 1
            return 1.0 / rank
        except ValueError:
            # If true_id is not in the list
            return 0.0

    def evaluate(
        self,
        k_values: List[int] = [1, 5, 10, 50, 100],
        initial_k: int = 100,
        rerank_k: int = 5,
        alpha: float = 0.7,
        num_samples: int = None,
        debug: bool = True,
        dataset_name: str = "code-rag-bench/mbpp" # "code-rag-bench/mbpp" or "openai_humaneval"
    ) -> Dict[str, Dict[int, float]]:
        dataset = load_dataset(dataset_name)

        query_name = 'prompt' if dataset_name == "openai_humaneval" else 'text'
        key = 'train' if dataset_name == "code-rag-bench/mbpp" else 'test'
        
        queries = [item[query_name] for item in dataset[key]]
        task_ids = [item['task_id'] for item in dataset[key]]
        
        if num_samples:
            queries = queries[:num_samples]
            task_ids = task_ids[:num_samples]
        
        if debug:
            print(f"\nEvaluating {len(queries)} queries")
            print("Sample task IDs to find:", task_ids[:5])
        
        baseline_recalls = {k: 0 for k in k_values}
        reranked_recalls = {k: 0 for k in k_values}

        baseline_position_scores = {k: 0.0 for k in k_values}
        reranked_position_scores = {k: 0.0 for k in k_values}
        
        # Add NDCG metrics
        baseline_ndcg = {k: 0.0 for k in k_values}
        reranked_ndcg = {k: 0.0 for k in k_values}
        
        # Add MRR metrics
        baseline_mrr = 0.0
        reranked_mrr = 0.0
        
        # Store the retrieved documents for each query
        retrieved_docs_list = []
        
        for query_idx, (query, true_id) in enumerate(tqdm(zip(queries, task_ids), total=len(queries))):
            if debug:
                print(f"\nProcessing query {query_idx + 1}/{len(queries)}")
                print(f"True task ID: {true_id}")
            
            results = self.retrieve_and_rerank(
                query=query,
                initial_k=initial_k,
                rerank_k=rerank_k,
                alpha=alpha,
                debug=debug
            )
            
            baseline_ids = [code.task_id for code in sorted(
                results,
                key=lambda x: x.initial_score,
                reverse=True
            )]
            
            reranked_ids = [code.task_id for code in results]

            true_id = str(true_id)
            baseline_ids = [str(id) for id in baseline_ids]
            reranked_ids = [str(id) for id in reranked_ids]
            
            # Store the retrieved documents for this query
            retrieved_docs_list.append({
                "query_id": query_idx,
                "query": query,
                "true_id": true_id,
                "baseline_docs": baseline_ids[:max(k_values)],
                "reranked_docs": reranked_ids[:max(k_values)]
            })
            
            if debug:
                print("\nTop 5 baseline IDs:", baseline_ids[:5])
                print("Top 5 reranked IDs:", reranked_ids[:5])
                print("Looking for true_id:", true_id)

            # Calculate NDCG for each k
            for k in k_values:
                # Calculate NDCG@k
                baseline_ndcg[k] += self._compute_ndcg_at_k(baseline_ids, true_id, k)
                reranked_ndcg[k] += self._compute_ndcg_at_k(reranked_ids, true_id, k)
                
                # Baseline position-aware metric
                try:
                    baseline_pos = baseline_ids[:k].index(true_id)
                    baseline_score = (k - baseline_pos) / k
                    baseline_position_scores[k] += baseline_score
                    if debug:
                        print(f"Baseline position-aware score @{k}: {baseline_score:.3f} (position {baseline_pos})")
                except ValueError:
                    if debug:
                        print(f"Baseline position-aware score @{k}: 0.000 (not found in top {k})")
                
                # Reranked position-aware metric
                try:
                    reranked_pos = reranked_ids[:k].index(true_id)
                    reranked_score = (k - reranked_pos) / k
                    reranked_position_scores[k] += reranked_score
                    if debug:
                        print(f"Reranked position-aware score @{k}: {reranked_score:.3f} (position {reranked_pos})")
                except ValueError:
                    if debug:
                        print(f"Reranked position-aware score @{k}: 0.000 (not found in top {k})")

                # Check if true_id is in the top-k results
                if true_id in baseline_ids[:k]:
                    baseline_recalls[k] += 1
                    if debug:
                        print(f"Found in baseline top-{k}")
                if true_id in reranked_ids[:k]:
                    reranked_recalls[k] += 1
                    if debug:
                        print(f"Found in reranked top-{k}")
            
            # Calculate MRR
            baseline_mrr += self._compute_mrr(baseline_ids, true_id)
            reranked_mrr += self._compute_mrr(reranked_ids, true_id)
        
        num_queries = len(queries)
        
        # Normalize metrics
        baseline_recalls = {k: count/num_queries for k, count in baseline_recalls.items()}
        reranked_recalls = {k: count/num_queries for k, count in reranked_recalls.items()}
        
        avg_baseline_position_scores = {k: score/num_queries for k, score in baseline_position_scores.items()}
        avg_reranked_position_scores = {k: score/num_queries for k, score in reranked_position_scores.items()}
        
        # Normalize NDCG
        avg_baseline_ndcg = {k: score/num_queries for k, score in baseline_ndcg.items()}
        avg_reranked_ndcg = {k: score/num_queries for k, score in reranked_ndcg.items()}
        
        # Normalize MRR
        avg_baseline_mrr = baseline_mrr / num_queries
        avg_reranked_mrr = reranked_mrr / num_queries
        
        if debug:
            print("\nAverage Metrics:")
            print(f"Baseline MRR: {avg_baseline_mrr:.4f}")
            print(f"Reranked MRR: {avg_reranked_mrr:.4f}")
            for k in k_values:
                print(f"Baseline NDCG@{k}: {avg_baseline_ndcg[k]:.4f}")
                print(f"Reranked NDCG@{k}: {avg_reranked_ndcg[k]:.4f}")
        
        return {
            "baseline": baseline_recalls,
            "reranked": reranked_recalls,
            "baseline-position-metric": avg_baseline_position_scores,
            "reranked-position-metric": avg_reranked_position_scores,
            "baseline-ndcg": avg_baseline_ndcg,
            "reranked-ndcg": avg_reranked_ndcg,
            "baseline-mrr": avg_baseline_mrr,
            "reranked-mrr": avg_reranked_mrr,
            "retrieved_docs": retrieved_docs_list
        }
import json
from typing import List, Dict, Tuple
from huggingface_hub import InferenceClient
from time import sleep
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class RetrievedCode:
    task_id: str
    code: str
    initial_score: float
    reranked_score: float = None

class CodeReranker:
    def __init__(
        self,
        hf_api_key: str,
        model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
        cache_dir: str = "reranking_cache"
    ):
        self.client = InferenceClient(api_key=hf_api_key)
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _generate_reranking_prompt(self, query: str, code: str) -> str:
        """Generate a prompt for the LLM to evaluate code relevance."""
        return f"""Given a programming task and a potential solution, evaluate how well the solution matches the requirements. Consider:
1. Does the solution address all requirements?
2. Is the implementation approach appropriate?
3. Are there any missing functionalities?

Programming Task:
{query}

Potential Solution:
{code}

Rate the relevance on a scale of 0-100 where:
0-20: Completely irrelevant
21-40: Barely relevant, missing major requirements
41-60: Partially relevant, missing some requirements
61-80: Mostly relevant, minor mismatches
81-100: Highly relevant, matches all requirements

Output only the numerical score. For example: 85"""

    def _get_relevance_score(self, query: str, code: str, max_retries: int = 3) -> float:
        """Get relevance score from LLM with retry logic."""
        prompt = self._generate_reranking_prompt(query, code)
        
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=10,
                    temperature=0.1
                )
                
                score_str = completion.choices[0].message.content.strip()
                try:
                    score = float(score_str)
                    return min(max(score, 0), 100)  # Ensure score is between 0 and 100
                except ValueError:
                    print(f"Failed to parse score: {score_str}")
                    return 50.0  # Default score if parsing fails
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to get relevance score: {e}")
                    return 50.0  # Default score after all retries
                wait_time = 2 ** attempt
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...")
                sleep(wait_time)
        
        return 50.0  # Default score if all retries fail

    def rerank_results(
        self,
        query: str,
        retrieved_codes: List[RetrievedCode],
        top_k: int = None,
        alpha: float = 0.5
    ) -> List[RetrievedCode]:
        """
        Rerank retrieved code using LLM-based relevance scoring.
        
        Args:
            query: Original programming task
            retrieved_codes: List of retrieved code snippets with initial scores
            top_k: Number of top results to rerank (to save API calls)
            alpha: Weight for combining initial and LLM scores (0: only initial, 1: only LLM)
        
        Returns:
            Reranked list of RetrievedCode objects
        """
        if top_k is None:
            top_k = len(retrieved_codes)
        
        # Sort by initial score and take top_k
        codes_to_rerank = sorted(
            retrieved_codes, 
            key=lambda x: x.initial_score, 
            reverse=True
        )[:top_k]
        
        print(f"Reranking top {top_k} results...")
        for code_item in tqdm(codes_to_rerank):
            # Get LLM-based relevance score
            relevance_score = self._get_relevance_score(query, code_item.code)
            
            # Normalize initial score to 0-100 range for combining
            normalized_initial = code_item.initial_score * 100
            
            # Combine scores using weighted average
            combined_score = (1 - alpha) * normalized_initial + alpha * relevance_score
            code_item.reranked_score = combined_score
        
        # Sort by reranked score
        reranked_results = sorted(
            codes_to_rerank, 
            key=lambda x: x.reranked_score, 
            reverse=True
        )
        
        # Add remaining results at the end if any
        if top_k < len(retrieved_codes):
            remaining = sorted(
                retrieved_codes[top_k:],
                key=lambda x: x.initial_score,
                reverse=True
            )
            reranked_results.extend(remaining)
        
        return reranked_results

    def evaluate_reranking(
        self,
        queries: List[str],
        retrieved_results: List[List[RetrievedCode]],
        true_ids: List[str],
        k_values: List[int] = [1, 5, 10],
        rerank_top_k: int = 10,
        alpha: float = 0.5
    ) -> Dict[str, Dict[int, float]]:
        """
        Evaluate reranking performance against baseline.
        
        Args:
            queries: List of programming tasks
            retrieved_results: List of retrieved codes for each query
            true_ids: List of correct task IDs for each query
            k_values: List of k values for computing Recall@k
            rerank_top_k: Number of top results to rerank
            alpha: Weight for combining scores
            
        Returns:
            Dictionary containing recall metrics for both baseline and reranked results
        """
        baseline_recalls = {k: 0 for k in k_values}
        reranked_recalls = {k: 0 for k in k_values}
        
        for query, results, true_id in tqdm(zip(queries, retrieved_results, true_ids), total=len(queries)):
            # Evaluate baseline
            baseline_ids = [code.task_id for code in results]
            for k in k_values:
                if true_id in baseline_ids[:k]:
                    baseline_recalls[k] += 1
            
            # Rerank and evaluate
            reranked = self.rerank_results(query, results, top_k=rerank_top_k, alpha=alpha)
            reranked_ids = [code.task_id for code in reranked]
            for k in k_values:
                if true_id in reranked_ids[:k]:
                    reranked_recalls[k] += 1
        
        # Normalize recalls
        num_queries = len(queries)
        baseline_recalls = {k: count/num_queries for k, count in baseline_recalls.items()}
        reranked_recalls = {k: count/num_queries for k, count in reranked_recalls.items()}
        
        return {
            "baseline": baseline_recalls,
            "reranked": reranked_recalls
        }

def main():
    import os
    from datasets import load_dataset
    
    # Initialize reranker
    hf_api_key = os.getenv("HF_API_KEY")
    if not hf_api_key:
        raise ValueError("Please set HF_API_KEY environment variable")
    
    reranker = CodeReranker(hf_api_key=hf_api_key)
    
    # Load HumanEval dataset
    dataset = load_dataset("openai_humaneval")
    print(dataset)
    print()
    print() 
    queries = [item['prompt'] for item in dataset['test']]
    true_ids = [item['task_id'] for item in dataset['test']]
    
    # Sample mock retrieved results (replace with actual retrieval results)
    mock_results = []
    for _ in queries:
        results = [
            RetrievedCode(
                task_id=f"task_{i}",
                code=f"def solution_{i}(): pass",
                initial_score=1.0 - (i * 0.1)
            )
            for i in range(10)
        ]
        mock_results.append(results)
    
    # Evaluate reranking
    k_values = [1, 5, 10]
    evaluation_results = reranker.evaluate_reranking(
        queries=queries[:5],  # Testing with first 5 queries
        retrieved_results=mock_results[:5],
        true_ids=true_ids[:5],
        k_values=k_values,
        rerank_top_k=5,
        alpha=0.7
    )
    
    # Print results
    print("\nEvaluation Results:")
    for method, recalls in evaluation_results.items():
        print(f"\n{method.title()} Recalls:")
        for k, recall in recalls.items():
            print(f"Recall@{k}: {recall:.3f}")

if __name__ == "__main__":
    main()
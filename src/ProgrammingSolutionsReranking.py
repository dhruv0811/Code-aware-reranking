import os 
from huggingface_hub import InferenceClient
from typing import List, Dict
from tqdm import tqdm
from dataclasses import dataclass
from time import sleep
import json
from datasets import load_dataset
import numpy as np

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

class ProgrammingSolutionsReranker:
    def __init__(
        self,
        corpus,
        llm_model_name: str = "meta-llama/Llama-3.1-70B-Instruct"
    ):
        hf_api_key = os.getenv("HF_API_KEY")
        if hf_api_key is None:
            raise ValueError("Must provide HuggingFace API key")
            
        self.corpus = corpus
        self.llm_model_name = llm_model_name
        self.client = InferenceClient(api_key=hf_api_key)


    def _generate_code_description(self, code: str, max_retries: int = 3) -> str:
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
                
                return completion.choices[0].message.content.strip()
                    
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
        
        if debug:
            print("\nSample retrieved task IDs:", [code.task_id for code in retrieved_codes[:5]])
        
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
            
            # Generate description of the code using LLM
            code_item.code_description = self._generate_code_description(code_item.code)
            
            if debug:
                print(f"Generated description: {code_item.code_description}")
            
            # Compute similarity between query and code description
            if code_item.code_description:
                description_similarity = self.corpus.compute_similarity(
                    query, 
                    code_item.code_description
                )
            else:
                description_similarity = 0.0
            
            # Combine initial similarity with description-based similarity
            normalized_initial = code_item.initial_score * 100
            code_item.reranked_score = (1 - alpha) * normalized_initial + alpha * (description_similarity * 100)
            
            if debug:
                print(f"Description similarity: {description_similarity:.4f}")
                print(f"Final reranked score: {code_item.reranked_score:.2f}")
        
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
        
        if debug:
            print("\nFinal reranking results:")
            for i, result in enumerate(reranked_results[:5]):
                print(f"Rank {i+1}: task_id={result.task_id}, score={result.reranked_score:.2f}")
                if result.code_description:
                    print(f"Description: {result.code_description}")
        
        return reranked_results

    def evaluate_humaneval(
        self,
        k_values: List[int] = [1, 5, 10],
        initial_k: int = 100,
        rerank_k: int = 5,
        alpha: float = 0.7,
        num_samples: int = None,
        debug: bool = True
    ) -> Dict[str, Dict[int, float]]:
        dataset = load_dataset("openai_humaneval")
        
        queries = [item['prompt'] for item in dataset['test']]
        task_ids = [item['task_id'] for item in dataset['test']]
        
        if num_samples:
            queries = queries[:num_samples]
            task_ids = task_ids[:num_samples]
        
        if debug:
            print(f"\nEvaluating {len(queries)} queries")
            print("Sample task IDs to find:", task_ids[:5])
        
        baseline_recalls = {k: 0 for k in k_values}
        reranked_recalls = {k: 0 for k in k_values}
        
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
            
            if debug:
                print("\nTop 5 baseline IDs:", baseline_ids[:5])
                print("Top 5 reranked IDs:", reranked_ids[:5])
                print("Looking for true_id:", true_id)
            
            for k in k_values:
                if true_id in baseline_ids[:k]:
                    baseline_recalls[k] += 1
                    if debug:
                        print(f"Found in baseline top-{k}")
                if true_id in reranked_ids[:k]:
                    reranked_recalls[k] += 1
                    if debug:
                        print(f"Found in reranked top-{k}")
        
        num_queries = len(queries)
        baseline_recalls = {k: count/num_queries for k, count in baseline_recalls.items()}
        reranked_recalls = {k: count/num_queries for k, count in reranked_recalls.items()}
        
        return {
            "baseline": baseline_recalls,
            "reranked": reranked_recalls
        }
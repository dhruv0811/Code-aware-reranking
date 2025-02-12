from pathlib import Path
import hashlib
import pickle
import os
from time import sleep
from typing import List, Dict, Optional
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import InferenceClient

PSEUDOCODE_PROMPT = """Given this programming task, break it down into clear pseudocode steps. 
Use standard pseudocode conventions and be specific about data structures and operations.

Task: {}

Provide only the pseudocode steps, with no additional information:"""

class CacheManager:
    def __init__(self, cache_dir: str = "pseudocode_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.modified = False

    def _generate_key(self, content: str, model_name: str) -> str:
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{content_hash}_{model_name}"

    def _get_cache_path(self, model_name: str) -> Path:
        safe_model_name = model_name.replace('/', '_')
        return self.cache_dir / f"cache_{safe_model_name}.pkl"

    def load(self, model_name: str) -> None:
        cache_path = self._get_cache_path(model_name)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"Loaded {len(self.cache)} entries from cache")
            except Exception as e:
                print(f"Cache loading failed: {e}")
                self.cache = {}
        else:
            self.cache = {}

    def save(self, model_name: str) -> None:
        if self.modified:
            cache_path = self._get_cache_path(model_name)
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.cache, f)
                print(f"Saved {len(self.cache)} entries to cache")
                self.modified = False
            except Exception as e:
                print(f"Cache saving failed: {e}")

    def get(self, content: str, model_name: str) -> Optional[str]:
        key = self._generate_key(content, model_name)
        return self.cache.get(key)

    def set(self, content: str, model_name: str, value: str) -> None:
        key = self._generate_key(content, model_name)
        self.cache[key] = value
        self.modified = True

class HumanEvalPseudoRetrieval:
    def __init__(
        self,
        corpus,
        llm_model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
        cache_dir: str = "pseudocode_cache"
    ):
        self.corpus = corpus
        self.model_name = llm_model_name
        api_key = os.getenv("HF_API_KEY")
        if not api_key:
            raise ValueError("HF_API_KEY environment variable not set")
        self.client = InferenceClient(api_key=api_key)
        self.cache_manager = CacheManager(cache_dir)
        self.cache_manager.load(llm_model_name)

    def generate_pseudocode(self, query: str, max_retries: int = 3) -> str:
        cached_pseudocode = self.cache_manager.get(query, self.model_name)
        if cached_pseudocode is not None:
            return cached_pseudocode

        prompt = PSEUDOCODE_PROMPT.format(query)
        
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.1
                )
                pseudocode = completion.choices[0].message.content.strip()
                
                self.cache_manager.set(query, self.model_name, pseudocode)
                self.cache_manager.save(self.model_name)
                
                return pseudocode
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Pseudocode generation failed: {e}")
                    return query
                wait_time = 2 ** attempt
                print(f"Retry {attempt + 1} in {wait_time}s...")
                sleep(wait_time)
        return query

    def evaluate(
        self,
        k_values: List[int] = [1, 5, 10, 50, 100],
        initial_k: int = 100,
        num_samples: Optional[int] = None,
        debug: bool = False,
        is_pseudo: bool = False,
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

        # Print the first query
        print("\nFirst Query:")
        print("-" * 80)
        print(queries[0])
        print("-" * 80)

        # Generate and print pseudocode for first query
        first_pseudocode = self.generate_pseudocode(queries[0])
        print("\nGenerated Pseudocode:")
        print("-" * 80)
        print(first_pseudocode)
        print("-" * 80)

        direct_recalls = {k: 0 for k in k_values}
        pseudo_recalls = {k: 0 for k in k_values}
        max_k = max(k_values)
        
        for query_idx, (query, true_id) in enumerate(tqdm(zip(queries, task_ids), total=len(queries))):
            pseudocode = self.generate_pseudocode(query)
            
            search_query = pseudocode if is_pseudo else query
            
            results = self.corpus.search(
                    search_query,
                    k=max_k
            )
                
            retrieved_ids = [doc.metadata['index'] for doc, _ in results]

            true_id = str(true_id)
            retrieved_ids = [str(id) for id in retrieved_ids]
                
            recalls = pseudo_recalls if is_pseudo else direct_recalls
            for k in k_values:
                if true_id in retrieved_ids[:k]:
                    recalls[k] += 1

        num_queries = len(queries)
        direct_recalls = {k: count/num_queries for k, count in direct_recalls.items()}
        pseudo_recalls = {k: count/num_queries for k, count in pseudo_recalls.items()}
        
        return {
            "baseline": direct_recalls,
            "pseudo": pseudo_recalls
        }
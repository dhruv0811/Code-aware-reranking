import os
import hashlib
import pickle
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
import numpy as np
import time
import torch

# Reuse the description prompt from Reranking.py
DESCRIPTION_PROMPT = """Analyze this code and provide a concise description of what it does.
Focus on the main functionality, algorithm, and purpose.
Keep the description under 100 words.

Code:
{}

Provide only the description with no additional text or formatting."""

class DescriptionCache:
    def __init__(self, cache_dir: str = "description_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.modified = False
    
    def _get_cache_key(self, code: str, model_name: str, norm_type: str) -> str:
        """Generate a unique key for the code-model-normalization combination."""
        # Use hash to handle potentially long code strings
        code_hash = hashlib.md5(code.encode()).hexdigest()
        return f"{code_hash}_{model_name}_{norm_type}"
    
    def _get_cache_file(self, model_name: str, norm_type: str) -> Path:
        """Get the cache file path for a specific model and normalization type."""
        safe_model_name = model_name.replace('/', '_')
        return self.cache_dir / f"desc_cache_{safe_model_name}_{norm_type}.pkl"
    
    def load_cache(self, model_name: str, norm_type: str):
        """Load the cache file for a specific model and normalization type."""
        cache_file = self._get_cache_file(model_name, norm_type)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"Loaded description cache from {cache_file} with {len(self.cache)} entries")
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.cache = {}
        else:
            self.cache = {}
    
    def save_cache(self, model_name: str, norm_type: str):
        """Save the cache to disk if it has been modified."""
        if self.modified:
            cache_file = self._get_cache_file(model_name, norm_type)
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
                print(f"Saved description cache to {cache_file} with {len(self.cache)} entries")
                self.modified = False
            except Exception as e:
                print(f"Error saving cache: {e}")
    
    def get(self, code: str, model_name: str, norm_type: str) -> Optional[str]:
        """Retrieve a cached description."""
        key = self._get_cache_key(code, model_name, norm_type)
        return self.cache.get(key)
    
    def set(self, code: str, model_name: str, norm_type: str, description: str):
        """Store a description in the cache."""
        key = self._get_cache_key(code, model_name, norm_type)
        self.cache[key] = description
        self.modified = True


class DescriptionBasedRetrieval:
    def __init__(
        self,
        corpus_dataset: str = "code-rag-bench/programming-solutions",
        embedding_model_name: str = "avsolatorio/GIST-large-Embedding-v0",
        llm_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        normalize_type: str = "none",
        cache_dir: str = "description_retrieval_cache"
    ):
        """Initialize the description-based retrieval system."""
        hf_api_key = os.getenv("HF_API_KEY")
        if hf_api_key is None:
            raise ValueError("Must provide HuggingFace API key as HF_API_KEY environment variable")
        
        print("Initializing Description-Based Retrieval System...")
        self.corpus_dataset = corpus_dataset
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.normalize_type = normalize_type
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )
        
        # Initialize LLM client for generating descriptions
        self.client = InferenceClient(api_key=hf_api_key)
        
        # Initialize cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.description_cache = DescriptionCache(str(self.cache_dir / "descriptions"))
        self.description_cache.load_cache(self.llm_model_name, self.normalize_type)
        
        # Load corpus
        self.corpus = load_dataset(corpus_dataset)
        
        # Initialize code normalizer
        from Corpus import CodeNormalizer
        self.code_normalizer = CodeNormalizer()
        
        # Will be initialized when build_index is called
        self.vectorstore = None
        self.code_map = {}  # Maps doc_id to original code
        
        print("Description-Based Retrieval System initialized!")
    
    def _generate_code_description(self, code: str, max_retries: int = 3) -> str:
        """Generate a description for a code snippet using the LLM."""
        # Check cache first
        cached_description = self.description_cache.get(code, self.llm_model_name, self.normalize_type)
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
                self.description_cache.set(code, self.llm_model_name, self.normalize_type, description)
                return description
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to get code description: {e}")
                    return ""
                wait_time = 2 ** attempt
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        return ""
    
    def _normalize_code(self, code: str, task: str = "humaneval") -> str:
        """Normalize code based on specified type."""
        if self.normalize_type == "none":
            return code
            
        # Use the normalizer from Corpus.py to maintain consistency
        from Corpus import ProgrammingSolutionsCorpus
        corpus_normalizer = ProgrammingSolutionsCorpus()
        return corpus_normalizer.normalize_code(code, self.normalize_type, task)
    
    def build_index(self, force_rebuild: bool = False):
        """Build the vector index of code descriptions."""
        # Check if index already exists
        index_path = self.cache_dir / f"index_{self.embedding_model_name}_{self.normalize_type}"
        if index_path.exists() and not force_rebuild:
            try:
                self.vectorstore = FAISS.load_local(str(index_path), self.embedding_model)
                
                # Load code map
                code_map_path = self.cache_dir / f"code_map_{self.normalize_type}.pkl"
                with open(code_map_path, 'rb') as f:
                    self.code_map = pickle.load(f)
                
                print(f"Loaded existing index from {index_path} with {len(self.code_map)} documents")
                return
            except Exception as e:
                print(f"Error loading existing index: {e}, rebuilding...")
        
        print("Building descriptions index...")
        documents = []
        metadatas = []
        self.code_map = {}
        
        # Process each item in the corpus
        for split in self.corpus:
            for item in tqdm(self.corpus[split], desc=f"Processing {split}"):
                if "text" not in item or "meta" not in item:
                    continue
                
                doc_text = item["text"]
                task_name = item["meta"].get("task_name", "humaneval")
                task_id = item["meta"].get("task_id", "")
                
                if not task_id:
                    continue
                
                # Apply normalization
                normalized_code = self._normalize_code(doc_text, task=task_name)
                
                # Generate description
                description = self._generate_code_description(normalized_code)
                
                if not description:
                    print(f"Warning: Could not generate description for task {task_id}, skipping")
                    continue
                
                # Store the code in our map
                self.code_map[task_id] = normalized_code
                
                # Store description as the indexable document
                documents.append(description)
                metadatas.append({
                    'source': f"{task_name}_{task_id}",
                    'index': task_id,
                })
                
                # Save cache periodically
                if len(documents) % 50 == 0:
                    self.description_cache.save_cache(self.llm_model_name, self.normalize_type)
        
        # Final cache save
        self.description_cache.save_cache(self.llm_model_name, self.normalize_type)
        
        print(f"Creating vector store with {len(documents)} descriptions...")
        self.vectorstore = FAISS.from_texts(
            documents,
            self.embedding_model,
            metadatas=metadatas
        )
        
        # Save the index
        self.vectorstore.save_local(str(index_path))
        
        # Save the code map
        code_map_path = self.cache_dir / f"code_map_{self.normalize_type}.pkl"
        with open(code_map_path, 'wb') as f:
            pickle.dump(self.code_map, f)
        
        print(f"Index built and saved to {index_path} with {len(documents)} descriptions")
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Retrieve code snippets based on description similarity with the query.
        
        Args:
            query: The user query
            k: Number of results to return
            
        Returns:
            List of tuples containing (task_id, code_snippet, similarity_score)
        """
        if self.vectorstore is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        results = self.vectorstore.similarity_search_with_score(query, k)
        
        retrieved_items = []
        for doc, score in results:
            task_id = doc.metadata['index']
            if task_id in self.code_map:
                code = self.code_map[task_id]
                # Convert score to similarity (1.0 - distance)
                similarity = 1.0 - score
                retrieved_items.append((task_id, code, similarity))
            else:
                print(f"Warning: Code for task_id {task_id} not found in code_map")
        
        return retrieved_items
    
    def evaluate(
        self,
        k_values: List[int] = [1, 5, 10, 25, 50, 100],
        num_samples: int = None,
        dataset_name: str = "openai_humaneval"  # or "code-rag-bench/mbpp"
    ) -> Dict[str, Dict[int, float]]:
        """
        Evaluate the description-based retrieval on a dataset.
        
        Args:
            k_values: List of k values to evaluate for recall
            num_samples: Optional number of samples to evaluate (for testing)
            dataset_name: Name of the dataset to evaluate on
            
        Returns:
            Dict with evaluation metrics
        """
        dataset = load_dataset(dataset_name)
        
        query_name = 'prompt' if dataset_name == "openai_humaneval" else 'text'
        key = 'train' if dataset_name == "code-rag-bench/mbpp" else 'test'
        
        queries = [item[query_name] for item in dataset[key]]
        task_ids = [item['task_id'] for item in dataset[key]]
        
        if num_samples:
            queries = queries[:num_samples]
            task_ids = task_ids[:num_samples]
        
        print(f"\nEvaluating {len(queries)} queries")
        
        # Track metrics
        recalls = {k: 0 for k in k_values}
        ndcg_values = {k: 0.0 for k in k_values}
        mrr = 0.0
        
        # Track retrieved documents
        retrieved_docs_list = []
        
        for query_idx, (query, true_id) in enumerate(tqdm(zip(queries, task_ids), total=len(queries))):
            print(f"\nProcessing query {query_idx + 1}/{len(queries)}")
            print(f"True task ID: {true_id}")
            
            # Get maximum k value for retrieval
            max_k = max(k_values)
            
            # Retrieve using descriptions
            results = self.retrieve(query, k=max_k)
            retrieved_ids = [task_id for task_id, _, _ in results]
            
            # Convert to string for consistent comparison
            true_id = str(true_id)
            retrieved_ids = [str(id) for id in retrieved_ids]
            
            # Store retrieved documents
            retrieved_docs_list.append({
                "query_id": query_idx,
                "query": query,
                "true_id": true_id,
                "retrieved_docs": retrieved_ids
            })
            
            print("\nTop 5 retrieved IDs:", retrieved_ids[:5])
            print("Looking for true_id:", true_id)
            
            # Calculate MRR
            try:
                rank = retrieved_ids.index(true_id) + 1
                mrr += 1.0 / rank
                print(f"Found at rank {rank}, MRR contribution: {1.0/rank:.4f}")
            except ValueError:
                print("Not found in retrieved results")
            
            # Calculate recall and NDCG for each k
            for k in k_values:
                # Calculate recall
                if true_id in retrieved_ids[:k]:
                    recalls[k] += 1
                    print(f"Found in top-{k}")
                
                # Calculate NDCG@k (using binary relevance)
                relevance = [1 if doc_id == true_id else 0 for doc_id in retrieved_ids[:k]]
                dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
                idcg = 1.0  # Binary relevance: only one relevant document
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcg_values[k] += ndcg
        
        # Normalize metrics
        num_queries = len(queries)
        recalls = {k: count/num_queries for k, count in recalls.items()}
        ndcg_values = {k: value/num_queries for k, value in ndcg_values.items()}
        mrr = mrr / num_queries
        
        print("\nEvaluation Results:")
        print(f"MRR: {mrr:.4f}")
        for k in k_values:
            print(f"Recall@{k}: {recalls[k]:.4f}, NDCG@{k}: {ndcg_values[k]:.4f}")
        
        return {
            "recalls": recalls,
            "ndcg": ndcg_values,
            "mrr": mrr,
            "retrieved_docs": retrieved_docs_list
        }


def main():
    """Example usage of the description-based retrieval system."""
    # Check for API key
    if "HF_API_KEY" not in os.environ:
        raise ValueError("Please set HF_API_KEY environment variable")
    
    # Initialize retrieval system
    retriever = DescriptionBasedRetrieval(
        embedding_model_name="avsolatorio/GIST-large-Embedding-v0",
        llm_model_name="meta-llama/Llama-3.1-8B-Instruct",
        normalize_type="both"
    )
    
    # Build the index (or load if it exists)
    retriever.build_index()
    
    # Test retrieval
    query = "Write a function to check if a string is a palindrome"
    results = retriever.retrieve(query, k=5)
    
    print("\nTest Retrieval Results:")
    for task_id, code, score in results:
        print(f"Task ID: {task_id}, Score: {score:.4f}")
        print(f"Code snippet: {code[:150]}...\n")
    
    # Evaluate on a dataset
    # Uncomment to run evaluation
    # results = retriever.evaluate(k_values=[1, 5, 10, 25, 50, 100])
    
    # Save evaluation results
    # import json
    # with open("description_retrieval_results.json", 'w') as f:
    #     json.dump(results, f, indent=2)

if __name__ == "__main__":
    import torch  # Added here to ensure it's available
    main()
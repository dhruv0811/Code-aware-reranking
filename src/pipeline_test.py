import json
from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm.auto import tqdm
import os
from openai import OpenAI
from time import sleep
import pickle
from pathlib import Path

class CodeRetrievalEvaluator:
    def __init__(
        self, 
        embedding_model_name: str = "avsolatorio/GIST-large-Embedding-v0",
        hf_api_key: str = None,
        inference_endpoint: str = None,
        cache_dir: str = "pseudocode_cache"
    ):
        """Initialize the evaluator with embedding model and inference endpoint."""
        if hf_api_key is None or inference_endpoint is None:
            raise ValueError("Must provide both HuggingFace API key and inference endpoint")
            
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )
        
        # Initialize OpenAI client with custom endpoint
        self.client = OpenAI(
            base_url=inference_endpoint,
            api_key=hf_api_key
        )
        
        # Setup cache directory for pseudocode
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.vectorstore = None

    def generate_pseudocode(self, query: str) -> str:
        """Generate pseudocode steps for solving the given query."""
        messages = [
            {
                "role": "system",
                "content": "You are an expert programmer. Break down programming tasks into clear pseudocode steps."
            },
            {
                "role": "user",
                "content": f"""Given this programming task, break it down into clear pseudocode steps. 
                Use standard pseudocode conventions and be specific about data structures and operations.

Task: {query}

Provide the pseudocode steps:"""
            }
        ]

        max_retries = 5
        wait_time = 1
        
        for attempt in range(max_retries):
            try:
                chat_completion = self.client.chat.completions.create(
                    model="tgi",
                    messages=messages,
                    max_tokens=256,
                    temperature=0.1,
                    stream=True,
                    top_p=None,
                    seed=None,
                    frequency_penalty=None,
                    presence_penalty=None
                )
                
                generated_text = ""
                for message in chat_completion:
                    if message.choices[0].delta.content is not None:
                        generated_text += message.choices[0].delta.content
                
                return generated_text.strip()
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to generate pseudocode after {max_retries} attempts: {e}")
                    return query
                wait_time = 2 ** attempt
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...")
                sleep(wait_time)
        
        return query

    def load_and_index_corpus(self, corpusName="code-rag-bench/programming-solutions"):
        """Load and index the original documents without modification."""
        print("Loading datasets...")
        corpus = load_dataset(corpusName)
        
        print("Processing documents...")
        documents = []
        metadatas = []

        split = 'train'
        if 'programming-solutions' in corpusName:
            doc_key = 'text'
        elif 'library-documentation' in corpusName:
            doc_key = 'doc_content'
        else:
            raise ValueError("Unknown dataset")
        
        for idx, item in enumerate(corpus[split]):
            doc_text = f"{item[doc_key]}"
            documents.append(doc_text)

            if 'programming-solutions' in corpusName:
                idx = item["meta"]["task_id"]
                metadatas.append({
                    'source': f"programming-solutions_{idx}",  
                    'index': idx,
                })
            elif 'library-documentation' in corpusName:
                idx = item["doc_id"]
                metadatas.append({
                    'source': f"library-documentation_{idx}",  
                    'index': idx
                })
        
        print(f"Creating vector store with {len(documents)} documents...")
        self.vectorstore = FAISS.from_texts(
            documents,
            self.embedding_model,
            metadatas=metadatas
        )
        print("Indexing complete!")

    def get_dataset_queries(self, dataset_name: str) -> Tuple[List[str], List[Any]]:
        """Get queries and true document IDs for a dataset."""
        if dataset_name == "humaneval":
            dataset = load_dataset("openai_humaneval")
            queries = [item['prompt'] for item in dataset['test']]
            task_ids = [item['task_id'] for item in dataset['test']]
            return queries, task_ids
            
        elif dataset_name == "mbpp":
            dataset = load_dataset("google-research-datasets/mbpp", "full")
            queries = [str(item['text']) for item in dataset['test']]
            task_ids = [str(item['task_id']) for item in dataset['test']]
            return queries, task_ids
            
        elif dataset_name == "ds1000":
            dataset = load_dataset("code-rag-bench/ds1000")
            queries = [str(item['prompt']) for item in dataset['train']]
            true_docs = [[x["title"] for x in item['docs']] for item in dataset['train']]
            return queries, true_docs
            
        elif dataset_name == "odex":
            dataset = load_dataset("code-rag-bench/odex")
            queries = [str(item['prompt']) for item in dataset['train']]
            true_docs = [[x["title"] for x in item['docs']] for item in dataset['train']]
            return queries, true_docs
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def generate_and_cache_pseudocode(self, dataset_name: str) -> Dict[str, str]:
        """Generate pseudocode for all queries in a dataset with per-query caching."""
        cache_file = self.cache_dir / f"{dataset_name}_pseudocode.pkl"
        
        # Initialize or load existing cache
        if cache_file.exists():
            print(f"Loading existing pseudocode cache for {dataset_name}...")
            with open(cache_file, 'rb') as f:
                pseudocode_map = pickle.load(f)
        else:
            print(f"Creating new pseudocode cache for {dataset_name}...")
            pseudocode_map = {}
        
        print(f"Generating pseudocode for {dataset_name} queries...")
        queries, _ = self.get_dataset_queries(dataset_name)
        
        for query in tqdm(queries):
            if query in pseudocode_map:
                continue  # Skip if already cached
                
            pseudocode = self.generate_pseudocode(query)
            pseudocode_map[query] = pseudocode
            
            # Cache after each new generation
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(pseudocode_map, f)
            except Exception as e:
                print(f"Warning: Failed to update cache after query: {e}")
        
        return pseudocode_map
    
    def get_cache_status(self, dataset_name: str) -> Dict[str, int]:
        """Get statistics about the pseudocode cache for a dataset."""
        cache_file = self.cache_dir / f"{dataset_name}_pseudocode.pkl"
        queries, _ = self.get_dataset_queries(dataset_name)
        total_queries = len(queries)
        
        if not cache_file.exists():
            return {
                "total_queries": total_queries,
                "cached_queries": 0,
                "remaining_queries": total_queries
            }
        
        with open(cache_file, 'rb') as f:
            pseudocode_map = pickle.load(f)
        
        cached_queries = len(pseudocode_map)
        return {
            "total_queries": total_queries,
            "cached_queries": cached_queries,
            "remaining_queries": total_queries - cached_queries
        }

    def evaluate_dataset(
        self,
        dataset_name: str,
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[int, float]:
        """Evaluate retrieval performance using pseudocode queries."""
        if not self.vectorstore:
            raise ValueError("Must call load_and_index_corpus first")

        # Get queries and true document ID
        queries, true_docs = self.get_dataset_queries(dataset_name)
        
        # Generate/load pseudocode for queries TATTIS
        pseudocode_map = self.generate_and_cache_pseudocode(dataset_name)
        
        recalls = {k: 0 for k in k_values}
        max_k = max(k_values)
        
        print(f"Evaluating {len(queries)} queries...")
        for query, true_doc in tqdm(zip(queries, true_docs), total=len(queries)):
            # Use pseudocode version for retrieval
            pseudocode_query = pseudocode_map[query]
            results = self.vectorstore.similarity_search_with_score(
                pseudocode_query,
                k=max_k
            )
            
            retrieved_ids = [doc.metadata['index'] for doc, _ in results]
            
            # Handle both single and multiple true document cases
            true_doc_list = [true_doc] if isinstance(true_doc, (str, int)) else true_doc
            
            for k in k_values:
                for doc_id in true_doc_list:
                    if doc_id in retrieved_ids[:k]:
                        recalls[k] += 1
                        break
        
        for k in k_values:
            recalls[k] = recalls[k] / len(queries)
            
        return recalls

def main():
    print("Initializing evaluator...")
    hf_api_key = os.getenv("HF_API_KEY")
    inference_endpoint = "https://xyf84eq3d6eusaui.us-east-1.aws.endpoints.huggingface.cloud/v1/"
    
    if not hf_api_key or not inference_endpoint:
        raise ValueError("Please set both HF_API_KEY and HF_INFERENCE_ENDPOINT environment variables")
        
    evaluator = CodeRetrievalEvaluator(
        hf_api_key=hf_api_key,
        inference_endpoint=inference_endpoint
    )
    
    # Load and index original documents
    print("Loading and indexing corpus...")
    evaluator.load_and_index_corpus(corpusName="code-rag-bench/library-documentation")
    
    # Evaluate each dataset
    datasets = ["ds1000"]
    k_values = [1, 5, 10, 50, 100]
    
    for dataset in datasets:
        print(f"\nEvaluating {dataset}...")
        cache_status = evaluator.get_cache_status(dataset)
        print(f"Cache status: {cache_status}")

        recalls = evaluator.evaluate_dataset(dataset, k_values)
        
        print(f"\n{dataset} Results:")
        for k, recall in recalls.items():
            print(f"Recall@{k}: {recall:.3f}")
        
        cache_status = evaluator.get_cache_status(dataset)
        print(f"Cache status: {cache_status}")

if __name__ == "__main__":
    main()
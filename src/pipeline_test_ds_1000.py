import json
from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import os
from huggingface_hub import InferenceClient
from time import sleep
import pickle
from pathlib import Path

class DS1000Retriever:
    def __init__(
        self, 
        embedding_model_name: str = "avsolatorio/GIST-large-Embedding-v0",
        llm_model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
        hf_api_key: str = None,
        cache_dir: str = "pseudocode_cache"
    ):
        if hf_api_key is None:
            raise ValueError("Must provide HuggingFace API key")
            
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )
        
        self.client = InferenceClient(api_key=hf_api_key)
        self.llm_model_name = llm_model_name
        
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

Provide only the pseudocode steps, with no additional information:"""
            }
        ]

        max_retries = 5
        wait_time = 1
        
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=messages,
                    max_tokens=256,
                    temperature=0.1
                )
                return completion.choices[0].message.content.strip()
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to generate pseudocode after {max_retries} attempts: {e}")
                    return query
                wait_time = 2 ** attempt
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...")
                sleep(wait_time)
        
        return query

    def load_and_index_corpus(self):
        """Load and index the library documentation corpus."""
        print("Loading library documentation dataset...")
        corpus = load_dataset("code-rag-bench/library-documentation")
        
        print("Processing documents...")
        documents = []
        metadatas = []
        
        for item in corpus['train']:
            doc_text = item["doc_content"]
            doc_id = item["doc_id"]
            metadatas.append({
                'source': f"library-documentation_{doc_id}",
                'index': doc_id
            })
            documents.append(doc_text)
        
        print(f"Creating vector store with {len(documents)} documents...")
        self.vectorstore = FAISS.from_texts(
            documents,
            self.embedding_model,
            metadatas=metadatas
        )
        print("Indexing complete!")

    def get_ds1000_queries(self) -> Tuple[List[str], List[List[str]]]:
        """Get DS1000 queries and their corresponding true document IDs."""
        dataset = load_dataset("code-rag-bench/ds1000")
        queries = [str(item['prompt']) for item in dataset['train']]
        true_docs = [[x["title"] for x in item['docs']] for item in dataset['train']]
        return queries, true_docs

    def generate_and_cache_pseudocode(self) -> Dict[str, str]:
        """Generate pseudocode for DS1000 queries with caching."""
        cache_file = self.cache_dir / "ds1000_pseudocode.pkl"
        
        if cache_file.exists():
            print("Loading existing pseudocode cache...")
            with open(cache_file, 'rb') as f:
                pseudocode_map = pickle.load(f)
        else:
            print("Creating new pseudocode cache...")
            pseudocode_map = {}
        
        print("Generating pseudocode for DS1000 queries...")
        queries, _ = self.get_ds1000_queries()
        
        for query in tqdm(queries):
            if query in pseudocode_map:
                continue
                
            pseudocode = self.generate_pseudocode(query)
            pseudocode_map[query] = pseudocode
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(pseudocode_map, f)
            except Exception as e:
                print(f"Warning: Failed to update cache after query: {e}")
        
        return pseudocode_map

    def get_cache_status(self) -> Dict[str, int]:
        """Get pseudocode cache statistics."""
        cache_file = self.cache_dir / "ds1000_pseudocode.pkl"
        queries, _ = self.get_ds1000_queries()
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

    def evaluate(
        self,
        k_values: List[int] = [1, 5, 10],
        use_pseudocode: bool = True,
        num_samples: int = None
    ) -> Dict[int, float]:
        """Evaluate retrieval performance with or without pseudocode."""
        if not self.vectorstore:
            raise ValueError("Must call load_and_index_corpus first")

        queries, true_docs = self.get_ds1000_queries()
        
        if num_samples:
            queries = queries[:num_samples]
            true_docs = true_docs[:num_samples]
        
        if use_pseudocode:
            pseudocode_map = self.generate_and_cache_pseudocode()
        
        recalls = {k: 0 for k in k_values}
        max_k = max(k_values)
        
        print(f"Evaluating {len(queries)} queries...")
        for query, true_doc_list in tqdm(zip(queries, true_docs), total=len(queries)):
            # Use pseudocode or original query
            search_query = pseudocode_map[query] if use_pseudocode else query
            
            results = self.vectorstore.similarity_search_with_score(
                search_query,
                k=max_k
            )
            
            retrieved_ids = [doc.metadata['index'] for doc, _ in results]
            
            for k in k_values:
                for doc_id in true_doc_list:
                    if doc_id in retrieved_ids[:k]:
                        recalls[k] += 1
                        break
        
        for k in k_values:
            recalls[k] = recalls[k] / len(queries)
            
        return recalls

def main():
    print("Initializing retriever...")
    hf_api_key = os.getenv("HF_API_KEY")
    
    if not hf_api_key:
        raise ValueError("Please set HF_API_KEY environment variable")
        
    retriever = DS1000Retriever(
        hf_api_key=hf_api_key,
        llm_model_name="meta-llama/Llama-3.1-70B-Instruct"
    )
    
    # Load and index library documentation
    print("Loading and indexing library documentation...")
    retriever.load_and_index_corpus()
    
    # Show cache status
    cache_status = retriever.get_cache_status()
    print(f"Cache status: {cache_status}")
    
    # Evaluate with and without pseudocode
    k_values = [1, 5, 10, 50, 100]
    num_samples = None  # Set to a number for testing with subset
    
    # Direct retrieval
    print("\nEvaluating direct retrieval...")
    direct_recalls = retriever.evaluate(
        k_values=k_values,
        use_pseudocode=False,
        num_samples=num_samples
    )
    
    # Pseudocode retrieval
    print("\nEvaluating pseudocode retrieval...")
    pseudo_recalls = retriever.evaluate(
        k_values=k_values,
        use_pseudocode=True,
        num_samples=num_samples
    )
    
    # Print results
    print("\nResults:")
    print("\nDirect Retrieval:")
    for k, recall in direct_recalls.items():
        print(f"Recall@{k}: {recall:.3f}")
        
    print("\nPseudocode Retrieval:")
    for k, recall in pseudo_recalls.items():
        print(f"Recall@{k}: {recall:.3f}")
    
    # Save results
    results = {
        "direct": direct_recalls,
        "pseudocode": pseudo_recalls
    }
    
    with open("results/ds1000_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
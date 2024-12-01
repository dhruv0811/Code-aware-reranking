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

class CodeRetrievalEvaluator:
    def __init__(
        self, 
        embedding_model_name: str = "avsolatorio/GIST-large-Embedding-v0",
        hf_api_key: str = None,
        inference_endpoint: str = None
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
        
        self.vectorstore = None
        self.corpus_documents = None

    def generate_pseudocode(self, query: str) -> str:
        """Generate pseudocode steps for solving the given query using custom inference endpoint."""
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

        # Make API request with exponential backoff
        max_retries = 5
        wait_time = 1
        
        for attempt in range(max_retries):
            try:
                # Create chat completion with streaming
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
                
                # Collect streamed response
                generated_text = ""
                for message in chat_completion:
                    if message.choices[0].delta.content is not None:
                        generated_text += message.choices[0].delta.content
                
                return generated_text.strip()
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to generate pseudocode after {max_retries} attempts: {e}")
                    return query  # Return original query as fallback
                wait_time = 2 ** attempt
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...")
                sleep(wait_time)
        
        return query  # Return original query if all attempts fail

    def rerank_results(
        self,
        query: str,
        initial_results: List[Tuple],
        k: int
    ) -> List[Tuple]:
        """
        Rerank the initial results using pseudocode similarity.
        
        Args:
            query: Original query
            initial_results: List of (document, score) tuples from initial retrieval
            k: Number of results to return after reranking
            
        Returns:
            List of reranked (document, score) tuples
        """
        # Generate pseudocode for the query
        query_pseudocode = self.generate_pseudocode(query)
        
        # Get embedding for query pseudocode
        query_embedding = self.embedding_model.embed_query(query_pseudocode)
        
        # Calculate similarities with retrieved documents
        reranked_results = []
        for doc, initial_score in initial_results:
            # Generate pseudocode for the document
            doc_pseudocode = self.generate_pseudocode(doc.page_content)
            
            # Get document pseudocode embedding
            doc_embedding = self.embedding_model.embed_query(doc_pseudocode)
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding)
            
            # Combine with initial score (you can adjust the weighting)
            final_score = 0.7 * similarity + 0.3 * initial_score
            
            reranked_results.append((doc, final_score))
        
        # Sort by final score and return top k
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        return reranked_results[:k]

    def evaluate_with_reranking_multiple_docs(
        self,
        queries: List[str],
        true_docs: List[List[str]],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[int, float]:
        """Generic evaluation method with reranking for multiple true documents per query."""
        recalls = {k: 0 for k in k_values}
        max_k = max(k_values)
        
        print(f"Evaluating {len(queries)} queries with reranking...")
        for query, true_doc_list in tqdm(zip(queries, true_docs), total=len(queries)):
            # Initial retrieval with larger k for reranking
            initial_results = self.vectorstore.similarity_search_with_score(
                query,
                k=100  # Retrieve more documents for reranking
            )
            
            # Rerank results
            reranked_results = self.rerank_results(query, initial_results, max_k)
            
            retrieved_task_ids = [doc.metadata['index'] for doc, _ in reranked_results]
            
            for k in k_values:
                for task_id in true_doc_list:
                    if task_id in retrieved_task_ids[:k]:
                        recalls[k] += 1
        
        for k in k_values:
            recalls[k] = recalls[k] / len(queries)
            
        return recalls

    def evaluate_humaneval(self, k_values: List[int] = [1, 5, 10]) -> Dict[int, float]:
        """Evaluate retrieval performance on HumanEval dataset with reranking."""
        if not self.vectorstore:
            raise ValueError("Must call load_and_index_corpus first")

        print("Loading HumanEval dataset for evaluation...")
        humaneval = load_dataset("openai_humaneval")
        
        queries = [item['prompt'] for item in humaneval['test']]
        task_ids = [item['task_id'] for item in humaneval['test']]
        
        return self.evaluate_with_reranking(queries, task_ids, k_values)

    def evaluate_mbpp(self, k_values: List[int] = [1, 5, 10]) -> Dict[int, float]:
        """Evaluate retrieval performance on MBPP dataset with reranking."""
        if not self.vectorstore:
            raise ValueError("Must call load_and_index_corpus first")

        print("Loading MBPP dataset for evaluation...")
        mbpp = load_dataset("google-research-datasets/mbpp", "full")
        
        queries = [str(item['text']) for item in mbpp['test']]
        task_ids = [str(item['task_id']) for item in mbpp['test']]
        
        return self.evaluate_with_reranking(queries, task_ids, k_values)

    def evaluate_ds1000(self, k_values: List[int] = [1, 5, 10]) -> Dict[int, float]:
        """Evaluate retrieval performance on DS-1000 dataset with reranking."""
        if not self.vectorstore:
            raise ValueError("Must call load_and_index_corpus first")

        print("Loading DS-1000 dataset for evaluation...")
        ds1000 = load_dataset("code-rag-bench/ds1000")
        
        queries = []
        true_docs = []
        
        for item in ds1000['train']:
            queries.append(str(item['prompt']))
            true_docs.append([x["title"] for x in item['docs']])
        
        return self.evaluate_with_reranking_multiple_docs(queries, true_docs, k_values)

    def evaluate_odex(self, k_values: List[int] = [1, 5, 10]) -> Dict[int, float]:
        """Evaluate retrieval performance on ODEX dataset with reranking."""
        if not self.vectorstore:
            raise ValueError("Must call load_and_index_corpus first")

        print("Loading ODEX dataset for evaluation...")
        odex = load_dataset("code-rag-bench/odex")
        
        queries = []
        true_docs = []
        
        for item in odex['train']:
            queries.append(str(item['prompt']))
            true_docs.append([x["title"] for x in item['docs']])
        
        return self.evaluate_with_reranking_multiple_docs(queries, true_docs, k_values)

    def load_and_index_corpus(self, corpusName="code-rag-bench/programming-solutions"):
        """Load the programming solutions dataset and create embeddings."""
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
            else:
                raise ValueError("Unknown dataset")
        
        print(f"Creating vector store with {len(documents)} documents...")
        self.vectorstore = FAISS.from_texts(
            documents,
            self.embedding_model,
            metadatas=metadatas
        )
        
        self.corpus_documents = list(zip(documents, metadatas))
        print("Indexing complete!")

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
    
    print("Loading and indexing corpus...")
    evaluator.load_and_index_corpus(corpusName="code-rag-bench/library-documentation")
    
    print("Evaluating on DS1000...")
    recalls = evaluator.evaluate_ds1000(k_values=[1, 5, 10, 50, 100])
    
    print("\nResults:")
    for k, recall in recalls.items():
        print(f"Recall@{k}: {recall:.3f}")

if __name__ == "__main__":
    main()
from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Any
import numpy as np
from tqdm.auto import tqdm
import os

class CodeRetrievalEvaluator:
    def __init__(self, embedding_model_name: str = "avsolatorio/GIST-large-Embedding-v0"):
        """Initialize the evaluator with a specific embedding model."""
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )
        self.vectorstore = None
        self.corpus_documents = None
        
    def load_and_index_corpus(self):
        """Load the programming solutions dataset and create embeddings."""
        print("Loading datasets...")
        corpus = load_dataset("code-rag-bench/programming-solutions")
        humaneval = load_dataset("openai_humaneval")
        
        print("Creating task ID mapping...")
        self.task_to_solution = {
            item['task_id']: item['canonical_solution'] 
            for item in humaneval['test']
        }
        
        print("Processing documents...")
        documents = []
        metadatas = []
        
        for idx, item in enumerate(corpus['train']):
            doc_text = f"{item['text']}"
            documents.append(doc_text)
            
            metadatas.append({
                'source': f"task_{idx}",  # Using a simple task ID format
                'index': item["meta"]["task_id"],
            })
        
        print(f"Creating vector store with {len(documents)} documents...")
        self.vectorstore = FAISS.from_texts(
            documents,
            self.embedding_model,
            metadatas=metadatas
        )
        
        self.corpus_documents = list(zip(documents, metadatas))
        print("Indexing complete!")

    def evaluate_humaneval(self, k_values: List[int] = [1, 5, 10]) -> Dict[int, float]:
        """Evaluate retrieval performance on HumanEval dataset using canonical solutions."""
        if not self.vectorstore:
            raise ValueError("Must call load_and_index_corpus first")

        print("Loading HumanEval dataset for evaluation...")
        humaneval = load_dataset("openai_humaneval")
        
        queries = []
        task_ids = []
        
        for item in humaneval['test']:
            queries.append(item['prompt'])
            task_ids.append(item['task_id'])
        
        recalls = {k: 0 for k in k_values}
        max_k = max(k_values)
        
        print(f"Evaluating {len(queries)} queries...")
        for query, task_id in tqdm(zip(queries, task_ids), total=len(queries)):

            results = self.vectorstore.similarity_search_with_score(
                query,
                k=max_k
            )
            
            retrieved_task_ids = [doc.metadata['index'] for doc, _ in results]

            if task_id == 'HumanEval/0':
                print(f"Query: {query}")
                print(f"Retrieved task IDs: {retrieved_task_ids}")
                print(f"Canonical solution: {self.task_to_solution[task_id]}")
                print()
            
            for k in k_values:
                if task_id in retrieved_task_ids[:k]:
                    recalls[k] += 1
        
        for k in k_values:
            recalls[k] = recalls[k] / len(queries)
            
        return recalls

def main():
    print("Initializing evaluator...")
    evaluator = CodeRetrievalEvaluator()
    
    print("Loading and indexing corpus...")
    evaluator.load_and_index_corpus()
    
    print("Evaluating on HumanEval...")
    recalls = evaluator.evaluate_humaneval(k_values=[1, 5, 10])
    
    print("\nResults:")
    for k, recall in recalls.items():
        print(f"Recall@{k}: {recall:.3f}")

if __name__ == "__main__":
    main()
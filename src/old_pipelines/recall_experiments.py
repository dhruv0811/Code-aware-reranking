import json
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
            
            for k in k_values:
                if task_id in retrieved_task_ids[:k]:
                    recalls[k] += 1
        
        for k in k_values:
            recalls[k] = recalls[k] / len(queries)
            
        return recalls

    def evaluate_mbpp(self, k_values: List[int] = [1, 5, 10]) -> Dict[int, float]:
        """Evaluate retrieval performance on MBPP dataset using canonical solutions."""
        if not self.vectorstore:
            raise ValueError("Must call load_and_index_corpus first")

        print("Loading MBPP dataset for evaluation...")
        mbpp = load_dataset("google-research-datasets/mbpp", "full")
        
        queries = []
        task_ids = []
        
        for item in mbpp['test']:
            queries.append(str(item['text']))
            task_ids.append(str(item['task_id']))
        
        recalls = {k: 0 for k in k_values}
        max_k = max(k_values)
        
        print(f"Evaluating {len(queries)} queries...")
        for query, task_id in tqdm(zip(queries, task_ids), total=len(queries)):

            results = self.vectorstore.similarity_search_with_score(
                query,
                k=max_k
            )
            
            retrieved_task_ids = [doc.metadata['index'] for doc, _ in results]
            
            for k in k_values:
                if task_id in retrieved_task_ids[:k]:
                    recalls[k] += 1
        
        for k in k_values:
            recalls[k] = recalls[k] / len(queries)
            
        return recalls
    
    def evaluate_ds1000(self, k_values: List[int] = [1, 5, 10]) -> Dict[int, float]:
        """Evaluate retrieval performance on DS-1000 dataset using canonical solutions."""
        if not self.vectorstore:
            raise ValueError("Must call load_and_index_corpus first")

        print("Loading DS-1000 dataset for evaluation...")
        ds1000 = load_dataset("code-rag-bench/ds1000")
        
        queries = []
        true_docs = []
        
        for item in ds1000['train']:
            queries.append(str(item['prompt']))
            true_docs.append([x["title"] for x in item['docs']])
        
        recalls = {k: 0 for k in k_values}
        max_k = max(k_values)
        
        print(f"Evaluating {len(queries)} queries...")
        count = 0
        for query, true_doc in tqdm(zip(queries, true_docs), total=len(queries)):

            results = self.vectorstore.similarity_search_with_score(
                query,
                k=max_k
            )
            
            retrieved_task_ids = [doc.metadata['index'] for doc, _ in results]

            # Calculate recall between true_doc and retrieved_task_ids
            # if len(true_doc) > 0:
            #     print(f"Query: {query}")
            #     print(f"True docs: {true_doc}")
            #     print(f"Retrieved docs: {retrieved_task_ids}")

            for k in k_values:
                for task_id in true_doc:
                    if task_id in retrieved_task_ids[:k]:
                        recalls[k] += 1
        
        for k in k_values:
            recalls[k] = recalls[k] / len(queries)
            
        return recalls
    
    def evaluate_odex(self, k_values: List[int] = [1, 5, 10]) -> Dict[int, float]:
        """Evaluate retrieval performance on ODEX dataset using canonical solutions."""
        if not self.vectorstore:
            raise ValueError("Must call load_and_index_corpus first")

        print("Loading DS-1000 dataset for evaluation...")
        ds1000 = load_dataset("code-rag-bench/odex")
        
        queries = []
        true_docs = []
        
        for item in ds1000['train']:
            queries.append(str(item['prompt']))
            true_docs.append([x["title"] for x in item['docs']])
        
        recalls = {k: 0 for k in k_values}
        max_k = max(k_values)
        
        print(f"Evaluating {len(queries)} queries...")
        count = 0
        for query, true_doc in tqdm(zip(queries, true_docs), total=len(queries)):

            results = self.vectorstore.similarity_search_with_score(
                query,
                k=max_k
            )
            
            retrieved_task_ids = [doc.metadata['index'] for doc, _ in results]

            # Calculate recall between true_doc and retrieved_task_ids
            # if len(true_doc) > 0:
            #     print(f"Query: {query}")
            #     print(f"True docs: {true_doc}")
            #     print(f"Retrieved docs: {retrieved_task_ids}")

            for k in k_values:
                for task_id in true_doc:
                    if task_id in retrieved_task_ids[:k]:
                        recalls[k] += 1
        
        for k in k_values:
            recalls[k] = recalls[k] / len(queries)
            
        return recalls

def main():
    print("Initializing evaluator...")
    evaluator = CodeRetrievalEvaluator()
    
    print("Loading and indexing corpus...")
    # evaluator.load_and_index_corpus()

    evaluator.load_and_index_corpus(corpusName="code-rag-bench/library-documentation")
    
    print("Evaluating on DS1000...")
    # recalls = evaluator.evaluate_humaneval(k_values=[1, 5, 10])
    # recalls = evaluator.evaluate_mbpp(k_values=[1, 5, 10])

    recalls = evaluator.evaluate_ds1000(k_values=[1, 5, 10, 50, 100])
    
    # recalls = evaluator.evaluate_odex(k_values=[1, 5, 10, 50, 100])
    
    print("\nResults:")
    for k, recall in recalls.items():
        print(f"Recall@{k}: {recall:.3f}")

if __name__ == "__main__":
    main()
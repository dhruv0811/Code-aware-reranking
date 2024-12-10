from datasets import load_dataset
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Any
from tqdm import tqdm
import os
from time import sleep
from dataclasses import dataclass
import re
import json

@dataclass
class RetrievedDoc:
    doc_id: str
    content: str
    initial_score: float
    reranked_score: float = None

class DS1000RetrieverReranker:
    def __init__(
        self,
        embedding_model_name: str = "avsolatorio/GIST-large-Embedding-v0",
        llm_model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
        hf_api_key: str = None
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
        self.vectorstore = None

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
        
        if documents:
            print("\nSample document:")
            print("Text:", documents[0][:200], "...")
            print("Metadata:", metadatas[0])
        
        print("Indexing complete!")

    def _generate_reranking_prompt(self, query: str, doc: str) -> str:
        return f"""Rate how well this library documentation matches the given programming task requirements.
Consider:
1. Relevance to the task
2. API functionality match
3. Usage example relevance
4. Implementation guidance

Task description:
{query}

Documentation:
{doc}

Rate relevance 0-100:
0-20: Completely irrelevant
21-40: Slightly relevant but not useful
41-60: Moderately relevant
61-80: Highly relevant
81-100: Perfect match

Output only the numerical score. Example: 85"""

    def _get_relevance_score(self, query: str, doc: str, max_retries: int = 3) -> float:
        prompt = self._generate_reranking_prompt(query, doc)
        
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=10,
                    temperature=0.1
                )
                
                score_str = completion.choices[0].message.content.strip()
                try:
                    score = float(score_str)
                    return min(max(score, 0), 100)
                except ValueError:
                    print(f"Failed to parse score: {score_str}")
                    return 50.0
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to get relevance score: {e}")
                    return 50.0
                wait_time = 2 ** attempt
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...")
                sleep(wait_time)
        
        return 50.0

    def retrieve_and_rerank(
        self,
        query: str,
        initial_k: int = 100,
        rerank_k: int = 5,
        alpha: float = 0.7,
        debug: bool = True
    ) -> List[RetrievedDoc]:
        if not self.vectorstore:
            raise ValueError("Must call load_and_index_corpus first")

        if debug:
            print(f"\nProcessing query: {query[:200]}...")
            
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=initial_k
        )
        
        if debug:
            print(f"Retrieved {len(results)} initial results")
            print("Sample retrieved scores:", [round(score, 4) for _, score in results[:5]])
        
        retrieved_docs = [
            RetrievedDoc(
                doc_id=doc.metadata['index'],
                content=doc.page_content,
                initial_score=1.0 - score  # Convert distance to similarity
            )
            for doc, score in results
        ]
        
        if debug:
            print("\nSample retrieved doc IDs:", [doc.doc_id for doc in retrieved_docs[:5]])
        
        docs_to_rerank = sorted(
            retrieved_docs,
            key=lambda x: x.initial_score,
            reverse=True
        )[:rerank_k]
        
        print(f"\nReranking top {rerank_k} results...")
        for doc_item in tqdm(docs_to_rerank):
            if debug:
                print(f"\nReranking doc_id: {doc_item.doc_id}")
                print(f"Initial score: {doc_item.initial_score:.4f}")
            
            relevance_score = self._get_relevance_score(query, doc_item.content)
            normalized_initial = doc_item.initial_score * 100
            doc_item.reranked_score = (1 - alpha) * normalized_initial + alpha * relevance_score
            
            if debug:
                print(f"LLM score: {relevance_score:.2f}")
                print(f"Final reranked score: {doc_item.reranked_score:.2f}")
        
        reranked_results = sorted(
            docs_to_rerank,
            key=lambda x: x.reranked_score,
            reverse=True
        )
        
        remaining = sorted(
            retrieved_docs[rerank_k:],
            key=lambda x: x.initial_score,
            reverse=True
        )
        reranked_results.extend(remaining)
        
        if debug:
            print("\nFinal reranking results:")
            for i, result in enumerate(reranked_results[:5]):
                print(f"Rank {i+1}: doc_id={result.doc_id}, score={result.reranked_score:.2f}")
        
        return reranked_results

    def evaluate_ds1000(
        self,
        k_values: List[int] = [1, 5, 10],
        initial_k: int = 100,
        rerank_k: int = 5,
        alpha: float = 0.7,
        num_samples: int = None,
        debug: bool = True
    ) -> Dict[str, Dict[int, float]]:
        dataset = load_dataset("code-rag-bench/ds1000")
        
        queries = [str(item['prompt']) for item in dataset['train']]
        true_docs = [[x["title"] for x in item['docs']] for item in dataset['train']]
        
        if num_samples:
            queries = queries[:num_samples]
            true_docs = true_docs[:num_samples]
        
        if debug:
            print(f"\nEvaluating {len(queries)} queries")
            print("Sample true docs to find:", true_docs[:5])
        
        baseline_recalls = {k: 0 for k in k_values}
        reranked_recalls = {k: 0 for k in k_values}
        
        for query_idx, (query, true_doc_list) in enumerate(tqdm(zip(queries, true_docs), total=len(queries))):
            if debug:
                print(f"\nProcessing query {query_idx + 1}/{len(queries)}")
                print(f"True doc IDs:", true_doc_list)
            
            results = self.retrieve_and_rerank(
                query=query,
                initial_k=initial_k,
                rerank_k=rerank_k,
                alpha=alpha,
                debug=debug
            )
            
            baseline_ids = [doc.doc_id for doc in sorted(
                results,
                key=lambda x: x.initial_score,
                reverse=True
            )]
            
            reranked_ids = [doc.doc_id for doc in results]
            
            if debug:
                print("\nTop 5 baseline IDs:", baseline_ids[:5])
                print("Top 5 reranked IDs:", reranked_ids[:5])
            
            for k in k_values:
                for true_doc in true_doc_list:
                    if true_doc in baseline_ids[:k]:
                        baseline_recalls[k] += 1
                        if debug:
                            print(f"Found in baseline top-{k}")
                        break
                    
                    if true_doc in reranked_ids[:k]:
                        reranked_recalls[k] += 1
                        if debug:
                            print(f"Found in reranked top-{k}")
                        break
        
        num_queries = len(queries)
        baseline_recalls = {k: count/num_queries for k, count in baseline_recalls.items()}
        reranked_recalls = {k: count/num_queries for k, count in reranked_recalls.items()}
        
        return {
            "baseline": baseline_recalls,
            "reranked": reranked_recalls
        }

def main():
    hf_api_key = os.getenv("HF_API_KEY")
    if not hf_api_key:
        raise ValueError("Please set HF_API_KEY environment variable")
    
    retriever_reranker = DS1000RetrieverReranker(hf_api_key=hf_api_key)
    
    # Load and index corpus
    print("\nLoading and indexing corpus...")
    retriever_reranker.load_and_index_corpus()
    
    # Print corpus statistics
    print("\nCorpus Statistics:")
    if retriever_reranker.vectorstore:
        print(f"Number of documents: {len(retriever_reranker.vectorstore.docstore._dict)}")
        sample_doc = next(iter(retriever_reranker.vectorstore.docstore._dict.values()))
        print("Sample document metadata:", sample_doc.metadata)
    
    # Evaluate
    print("\nEvaluating retrieval and reranking...")
    results = retriever_reranker.evaluate_ds1000(
        k_values=[1, 5, 10, 50, 100],
        initial_k=100,
        rerank_k=5,
        alpha=0.7,
        num_samples=5,  # Test with 5 samples first
        debug=True
    )
    
    # Print results
    print("\nFinal Evaluation Results:")
    for method, recalls in results.items():
        print(f"\n{method.title()} Recalls:")
        for k, recall in recalls.items():
            print(f"Recall@{k}: {recall:.3f}")
    
    # Save results
    with open("ds1000_reranking_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
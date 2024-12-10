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
import ast
import json


@dataclass
class RetrievedCode:
    task_id: str
    code: str
    initial_score: float
    reranked_score: float = None

class CodeRetrieverReranker:
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

    def remove_docstring(self, code: str) -> str:
        """Remove docstring from Python code while preserving other comments."""
        docstring_pattern = r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\''
        return re.sub(docstring_pattern, '', code)

    def load_and_index_corpus(self):
        """Load and index the programming solutions corpus."""
        print("Loading datasets...")
        corpus = load_dataset("code-rag-bench/programming-solutions")
        
        print("Processing documents...")
        documents = []
        metadatas = []
        
        for item in corpus['train']:
            if item["meta"]["task_name"] == "humaneval":
                doc_text = item["text"]
                doc_text = self.remove_docstring(doc_text)
                
                idx = item["meta"]["task_id"]
                metadatas.append({
                    'source': f"programming-solutions_{idx}",
                    'index': idx,
                })
                documents.append(doc_text)
        
        print(f"Creating vector store with {len(documents)} documents...")
        self.vectorstore = FAISS.from_texts(
            documents,
            self.embedding_model,
            metadatas=metadatas
        )
        
        # Print sample document for debugging
        if documents:
            print("\nSample document:")
            print("Text:", documents[0][:200], "...")
            print("Metadata:", metadatas[0])
        
        print("Indexing complete!")

    def _generate_reranking_prompt(self, query: str, code: str) -> str:
        return f"""Rate how well this code solution matches the given programming task requirements.
Consider:
1. Implementation completeness
2. Algorithm correctness
3. Edge case handling
4. Code efficiency

Task description:
{query}

Code solution:
{code}

Rate relevance 0-100:
0-20: Wrong/irrelevant
21-40: Missing major requirements
41-60: Partially complete
61-80: Mostly correct
81-100: Perfect match

Output only the numerical score. Example: 85"""

    def _get_relevance_score(self, query: str, code: str, max_retries: int = 3) -> float:
        prompt = self._generate_reranking_prompt(query, code)
        
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
    ) -> List[RetrievedCode]:
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
        
        retrieved_codes = [
            RetrievedCode(
                task_id=doc.metadata['index'],
                code=doc.page_content,
                initial_score=1.0 - score  # Convert distance to similarity
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
        
        print(f"\nReranking top {rerank_k} results...")
        for code_item in tqdm(codes_to_rerank):
            if debug:
                print(f"\nReranking code for task_id: {code_item.task_id}")
                print(f"Initial score: {code_item.initial_score:.4f}")
            
            relevance_score = self._get_relevance_score(query, code_item.code)
            normalized_initial = code_item.initial_score * 100
            code_item.reranked_score = (1 - alpha) * normalized_initial + alpha * relevance_score
            
            if debug:
                print(f"LLM score: {relevance_score:.2f}")
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

def main():
    hf_api_key = os.getenv("HF_API_KEY")
    if not hf_api_key:
        raise ValueError("Please set HF_API_KEY environment variable")
    
    retriever_reranker = CodeRetrieverReranker(hf_api_key=hf_api_key)
    
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
    results = retriever_reranker.evaluate_humaneval(
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
    with open("humaneval_reranking_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
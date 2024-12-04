import json
from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm.auto import tqdm
import os
from huggingface_hub import InferenceClient
from time import sleep
import pickle
from pathlib import Path
import re

class CodeRetrievalEvaluator:
    def __init__(
        self, 
        embedding_model_name: str = "avsolatorio/GIST-large-Embedding-v0",
        hf_api_key: str = None,
        model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
        cache_dir: str = "pseudocode_cache"
    ):
        """Initialize the evaluator with embedding model and inference client."""
        if hf_api_key is None:
            raise ValueError("Must provide HuggingFace API key")
            
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )
        
        self.client = InferenceClient(api_key=hf_api_key)
        self.model_name = model_name
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.vectorstore = None

    # [Previous normalization methods remain unchanged]
    def remove_docstring(self, code: str) -> str:
        """Remove docstring from Python code while preserving other comments."""
        docstring_pattern = r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\''
        return re.sub(docstring_pattern, '', code)

    def normalize_code(self, code: str, normalize_type: str = "none") -> str:
        """Normalize code based on specified type."""
        if normalize_type == "none":
            return code
            
        code = self.remove_docstring(code)
        
        if normalize_type == "variables":
            return self._replace_variables(code)
        elif normalize_type == "functions":
            return self._replace_function_names(code)
        elif normalize_type == "both":
            code = self._replace_function_names(code)
            return self._replace_variables(code)
        else:
            raise ValueError(f"Unknown normalization type: {normalize_type}")

    def _replace_variables(self, code: str) -> str:
        """Replace variable names with generic placeholders while preserving comments."""
        lines = code.split('\n')
        processed_lines = []
        var_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        var_map = {}
        var_counter = 0
        
        python_keywords = {'def', 'return', 'if', 'else', 'for', 'while', 'in', 'True', 'False', 
                         'None', 'and', 'or', 'not', 'is', 'len', 'range', 'print', 'sum', 'max', 
                         'min', 'sorted', 'list', 'dict', 'set', 'tuple', 'sort'}
        
        def replace_match(match):
            nonlocal var_counter
            var_name = match.group(0)
            if var_name in python_keywords:
                return var_name
            if var_name not in var_map:
                var_map[var_name] = f'var_{var_counter}'
                var_counter += 1
            return var_map[var_name]
        
        for line in lines:
            comment_start = line.find('#')
            if comment_start != -1:
                code_part = line[:comment_start]
                comment_part = line[comment_start:]
                processed_code = re.sub(var_pattern, replace_match, code_part)
                processed_lines.append(processed_code + comment_part)
            else:
                processed_lines.append(re.sub(var_pattern, replace_match, line))
        
        return '\n'.join(processed_lines)

    def _replace_function_names(self, code: str) -> str:
        """Replace function names with generic placeholders while preserving structure."""
        lines = code.split('\n')
        processed_lines = []
        func_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        func_map = {}
        func_counter = 0
        
        for line in lines:
            comment_start = line.find('#')
            if comment_start != -1:
                code_part = line[:comment_start]
                comment_part = line[comment_start:]
            else:
                code_part = line
                comment_part = ''
            
            def replace_func(match):
                nonlocal func_counter
                func_name = match.group(1)
                if func_name not in func_map:
                    func_map[func_name] = f'func_{func_counter}'
                    func_counter += 1
                return f'def {func_map[func_name]}'
            
            processed_code = re.sub(func_pattern, replace_func, code_part)
            
            for old_name, new_name in func_map.items():
                processed_code = re.sub(r'\b' + re.escape(old_name) + r'\b(?!\s*\()', new_name, processed_code)
            
            processed_lines.append(processed_code + comment_part)
        
        return '\n'.join(processed_lines)

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
                    model=self.model_name,
                    messages=messages,
                    max_tokens=500,
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

    def load_and_index_corpus(self, normalize_type: str = "none"):
        """Load and index the programming solutions corpus with optional normalization."""
        print("Loading datasets...")
        corpus = load_dataset("code-rag-bench/programming-solutions")
        
        print("Processing documents...")
        documents = []
        metadatas = []
        
        for item in corpus['train']:
            if item["meta"]["task_name"] == "humaneval":
                doc_text = item["text"]
                doc_text = self.normalize_code(doc_text, normalize_type)
                
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
        print("Indexing complete!")

    def generate_and_cache_pseudocode(self) -> Dict[str, str]:
        """Generate pseudocode for HumanEval queries with caching."""
        cache_file = self.cache_dir / "humaneval_pseudocode.pkl"
        
        if cache_file.exists():
            print("Loading existing pseudocode cache...")
            with open(cache_file, 'rb') as f:
                pseudocode_map = pickle.load(f)
        else:
            print("Creating new pseudocode cache...")
            pseudocode_map = {}
        
        print("Generating pseudocode for HumanEval queries...")
        dataset = load_dataset("openai_humaneval")
        queries = [item['prompt'] for item in dataset['test']]
        task_ids = [item['task_id'] for item in dataset['test']]
        
        for query, task_id in tqdm(zip(queries, task_ids), total=len(queries)):
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

    def evaluate(
        self,
        k_values: List[int] = [1, 5, 10],
        use_pseudocode: bool = True
    ) -> Dict[int, float]:
        """Evaluate retrieval performance on HumanEval dataset."""
        if not self.vectorstore:
            raise ValueError("Must call load_and_index_corpus first")

        dataset = load_dataset("openai_humaneval")
        queries = [item['prompt'] for item in dataset['test']]
        task_ids = [item['task_id'] for item in dataset['test']]
        
        if use_pseudocode:
            pseudocode_map = self.generate_and_cache_pseudocode()
        
        recalls = {k: 0 for k in k_values}
        max_k = max(k_values)
        
        print(f"Evaluating {len(queries)} queries...")
        for query, task_id in tqdm(zip(queries, task_ids), total=len(queries)):
            search_query = pseudocode_map[query] if use_pseudocode else query
            
            results = self.vectorstore.similarity_search_with_score(
                search_query,
                k=max_k
            )
            
            retrieved_ids = [doc.metadata['index'] for doc, _ in results]
            
            for k in k_values:
                if task_id in retrieved_ids[:k]:
                    recalls[k] += 1
        
        for k in k_values:
            recalls[k] = recalls[k] / len(queries)
            
        return recalls

def main():
    print("Initializing evaluator...")
    hf_api_key = os.getenv("HF_API_KEY")
    
    if not hf_api_key:
        raise ValueError("Please set HF_API_KEY environment variable")
        
    evaluator = CodeRetrievalEvaluator(
        hf_api_key=hf_api_key,
        model_name="meta-llama/Llama-3.1-70B-Instruct"
    )
    
    # Test different normalization strategies
    normalize_types = ["none", "variables", "functions", "both"]
    k_values = [1, 5, 10, 50, 100]
    
    results = {}
    
    for norm_type in normalize_types:
        print(f"\nTesting normalization type: {norm_type}")
        
        # Load and index corpus with current normalization
        evaluator.load_and_index_corpus(normalize_type=norm_type)
        
        # Evaluate with and without pseudocode
        for use_pseudo in [False, True]:
            test_name = f"{norm_type}_{'pseudo' if use_pseudo else 'direct'}"
            print(f"\nRunning evaluation: {test_name}")
            
            recalls = evaluator.evaluate(k_values=k_values, use_pseudocode=use_pseudo)
            results[test_name] = recalls
            
            print(f"\n{test_name} Results:")
            for k, recall in recalls.items():
                print(f"Recall@{k}: {recall:.3f}")
    
    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
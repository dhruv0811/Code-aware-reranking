import json
from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Any
import numpy as np
from tqdm.auto import tqdm
import os
import re

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

    def remove_docstring(self, code: str) -> str:
        """Remove docstring from Python code while preserving other comments."""
        # Pattern to match triple-quoted strings (both single and double quotes)
        docstring_pattern = r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\''
        
        # Remove docstrings
        code_without_docstring = re.sub(docstring_pattern, '', code)
        
        return code_without_docstring

    def replace_function_names_only(self, code: str) -> str:
        """Replace only function names with generic placeholders while preserving variables and comments."""
        lines = code.split('\n')
        processed_lines = []
        
        # Pattern to match function definitions
        func_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        
        # Keep track of function name mappings
        func_map = {}
        func_counter = 0
        
        for line in lines:
            # Check if the line contains a comment
            comment_start = line.find('#')
            if comment_start != -1:
                code_part = line[:comment_start]
                comment_part = line[comment_start:]
            else:
                code_part = line
                comment_part = ''
            
            # Replace function names in definitions
            def replace_func(match):
                nonlocal func_counter
                func_name = match.group(1)
                if func_name not in func_map:
                    func_map[func_name] = f'func_{func_counter}'
                    func_counter += 1
                return f'def {func_map[func_name]}'
            
            # Process the code part
            processed_code = re.sub(func_pattern, replace_func, code_part)
            
            # Replace function calls with their new names
            for old_name, new_name in func_map.items():
                # Look for function calls (word boundaries prevent partial matches)
                processed_code = re.sub(r'\b' + re.escape(old_name) + r'\b(?!\s*\()', new_name, processed_code)
            
            # Reconstruct the line
            processed_lines.append(processed_code + comment_part)
        
        return '\n'.join(processed_lines)

    def replace_variables_preserve_functions(self, code: str) -> str:
        """Replace variables while preserving function names and comments."""
        lines = code.split('\n')
        processed_lines = []
        
        # Define variable name pattern
        var_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        
        # Keep track of variable mappings
        var_map = {}
        var_counter = 0
        
        # Keep track of function names to preserve them
        function_names = set()
        func_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        for line in lines:
            func_match = re.search(func_pattern, line)
            if func_match:
                function_names.add(func_match.group(1))

        def replace_match(match):
            nonlocal var_counter
            var_name = match.group(0)
            
            # Skip Python keywords, built-in functions, and function names
            python_keywords = {'def', 'return', 'if', 'else', 'for', 'while', 'in', 'True', 'False', 
                             'None', 'and', 'or', 'not', 'is', 'len', 'range', 'print', 'sum', 'max', 
                             'min', 'sorted', 'list', 'dict', 'set', 'tuple', 'sort'}
            if var_name in python_keywords or var_name in function_names:
                return var_name
                
            # If we haven't seen this variable before, create a new mapping
            if var_name not in var_map:
                var_map[var_name] = f'var_{var_counter}'
                var_counter += 1
                
            return var_map[var_name]
        
        for line in lines:
            # Check if the line contains a comment
            comment_start = line.find('#')
            if comment_start != -1:
                code_part = line[:comment_start]
                comment_part = line[comment_start:]
                # Process only the code part
                processed_code = re.sub(var_pattern, replace_match, code_part)
                # Reconstruct the line with the original comment
                processed_lines.append(processed_code + comment_part)
            else:
                # Process the entire line if there's no comment
                processed_lines.append(re.sub(var_pattern, replace_match, line))
        
        return '\n'.join(processed_lines)
    
    def replace_variables(self, code: str) -> str:
        """Replace variable names with generic placeholders while preserving comments."""
        # First, let's split the code into lines and process each line
        lines = code.split('\n')
        processed_lines = []
        
        # Define common Python variable name pattern
        var_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        
        # Keep track of variable mappings
        var_map = {}
        var_counter = 0
        
        def replace_match(match):
            nonlocal var_counter
            var_name = match.group(0)
            
            # Skip Python keywords and built-in functions
            python_keywords = {'def', 'return', 'if', 'else', 'for', 'while', 'in', 'True', 'False', 
                             'None', 'and', 'or', 'not', 'is', 'len', 'range', 'print', 'sum', 'max', 
                             'min', 'sorted', 'list', 'dict', 'set', 'tuple', 'sort'}
            if var_name in python_keywords:
                return var_name
                
            # If we haven't seen this variable before, create a new mapping
            if var_name not in var_map:
                var_map[var_name] = f'var_{var_counter}'
                var_counter += 1
                
            return var_map[var_name]
        
        for line in lines:
            # Check if the line contains a comment
            comment_start = line.find('#')
            if comment_start != -1:
                # Split the line into code and comment
                code_part = line[:comment_start]
                comment_part = line[comment_start:]
                # Only process the code part
                processed_code = re.sub(var_pattern, replace_match, code_part)
                # Reconstruct the line with the original comment
                processed_lines.append(processed_code + comment_part)
            else:
                # Process the entire line if there's no comment
                processed_lines.append(re.sub(var_pattern, replace_match, line))
        
        return '\n'.join(processed_lines)
        
    def load_and_index_corpus(self, corpusName="code-rag-bench/programming-solutions"):
        """Load the programming solutions dataset and create embeddings."""
        print("Loading datasets...")
        corpus = load_dataset(corpusName)
        humaneval_dataset = load_dataset('code-rag-bench/humaneval')['train']
        
        print("Processing documents...")
        documents = []
        metadatas = []

        split = 'train'
        if 'programming-solutions' in corpusName:
            doc_key = 'text'
        else:
            raise ValueError("Unknown dataset")
        
        for idx, item in enumerate(corpus[split]):
            doc_text = f"{item[doc_key]}"
            

            if 'programming-solutions' in corpusName:
                idx = item["meta"]["task_id"]
                task_name = item["meta"]["task_name"]
                metadatas.append({
                    'source': f"programming-solutions_{idx}",  
                    'index': idx,
                })
                if task_name != "humaneval":
                    # print("Original Solution: ", doc_text)
                    doc_text = self.replace_variables(doc_text)
                    # print("Replaced Variable Solution: ", doc_text)
                    documents.append(doc_text)

                else:
                    matching_idx = humaneval_dataset['task_id'].index(idx)
                    # print("Doc Text: ", doc_text)
                    doc_text = self.remove_docstring(doc_text)
                    print("Comments Text: ", doc_text)
                    # doc_text = humaneval_dataset[matching_idx]['canonical_solution']
                    # print("Original Canonical Solution: ", doc_text)
                    doc_text = self.replace_variables(doc_text)
                    print("Replaced Variable Canonical Solution: ", doc_text)
                    # print("Replacing original document with ", doc_text)
                    documents.append(doc_text)
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

def main():
    print("Initializing evaluator...")
    evaluator = CodeRetrievalEvaluator()
    
    print("Loading and indexing corpus...")
    # evaluator.load_and_index_corpus()

    evaluator.load_and_index_corpus(corpusName="code-rag-bench/programming-solutions")
    
    print("Evaluating on HumanEval...")
    recalls = evaluator.evaluate_humaneval(k_values=[1, 5, 10, 50, 100])
    
    print("\nResults:")
    for k, recall in recalls.items():
        print(f"Recall@{k}: {recall:.3f}")

if __name__ == "__main__":
    main()
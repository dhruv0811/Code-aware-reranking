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
import ast


class VariableRenamer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.name_map: Dict[str, str] = {}
        # Track scopes to handle variable shadowing
        self.scopes: list[Set[str]] = [set()]
        # Names that shouldn't be renamed
        self.preserved_names = {
            # Built-in functions and types
            'len', 'range', 'print', 'sum', 'max', 'min', 'sorted', 'enumerate',
            'zip', 'map', 'filter', 'abs', 'all', 'any', 'round', 'pow', 'type',
            'list', 'dict', 'set', 'tuple', 'str', 'int', 'float', 'bool',
            # Common module imports
            'np', 'pd', 'plt', 're', 'os', 'sys', 'math',
            # Common methods
            'append', 'extend', 'insert', 'remove', 'pop', 'clear', 'index',
            'count', 'sort', 'reverse', 'copy', 'split', 'join', 'strip',
        }

    def enter_scope(self):
        self.scopes.append(set())

    def exit_scope(self):
        self.scopes.pop()

    def add_to_scope(self, name: str):
        self.scopes[-1].add(name)

    def is_in_current_scope(self, name: str) -> bool:
        return name in self.scopes[-1]

    def get_new_name(self, old_name: str) -> str:
        if old_name not in self.name_map:
            self.name_map[old_name] = f'var_{self.counter}'
            self.counter += 1
        return self.name_map[old_name]

    def should_rename(self, name: str) -> bool:
        return not (
            name.startswith('__') or  # Skip magic methods
            name in self.preserved_names or  # Skip preserved names
            name.startswith('var_')  # Skip already normalized names
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Handle function definitions and their arguments."""
        self.enter_scope()
        
        # Handle function arguments
        for arg in node.args.args:
            if self.should_rename(arg.arg):
                new_name = self.get_new_name(arg.arg)
                self.add_to_scope(new_name)
                arg.arg = new_name

        # Process function body
        node.body = [self.visit(n) for n in node.body]
        
        self.exit_scope()
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Handle variable names in expressions."""
        if isinstance(node.ctx, (ast.Store, ast.Load)) and self.should_rename(node.id):
            if isinstance(node.ctx, ast.Store):
                new_name = self.get_new_name(node.id)
                self.add_to_scope(new_name)
                node.id = new_name
            else:  # Load context
                if node.id in self.name_map:
                    node.id = self.name_map[node.id]
        return node

    def visit_Import(self, node: ast.Import) -> ast.Import:
        """Preserve import names."""
        for alias in node.names:
            self.preserved_names.add(alias.asname or alias.name)
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        """Preserve from-import names."""
        for alias in node.names:
            self.preserved_names.add(alias.asname or alias.name)
        return node

class CodeNormalizer:
    @staticmethod
    def normalize_variables(code: str) -> str:
        """
        Normalize variable names in Python code using AST.
        
        Args:
            code (str): Python source code
            
        Returns:
            str: Normalized Python code
        """
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Apply variable renaming
            renamer = VariableRenamer()
            transformed_tree = renamer.visit(tree)
            
            # Fix formatting to handle multi-line strings and comments
            ast.fix_missing_locations(transformed_tree)
            
            # Convert back to source code
            return ast.unparse(transformed_tree)
        except Exception as e:
            print(f"Error normalizing code: {e}")
            return code  # Return original code if normalization fails


class CodeRetrievalEvaluator:
    def __init__(
        self, 
        embedding_model_name: str = "avsolatorio/GIST-large-Embedding-v0",
        hf_api_key: str = None,
        model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
        cache_dir: str = "../pseudocode_humaneval"
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
    
    def _normalize_variables_ast(self, code: str) -> str:
        """Normalize variable names in Python code using AST."""
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Apply variable renaming
            renamer = VariableRenamer()
            transformed_tree = renamer.visit(tree)
            
            # Fix formatting
            ast.fix_missing_locations(transformed_tree)
            
            # Convert back to source code
            return ast.unparse(transformed_tree)
        except Exception as e:
            print(f"Warning: AST normalization failed - {e}")
            return code  # Return original code if normalization fails

    def normalize_code(self, code: str, normalize_type: str = "none") -> str:
        """Normalize code based on specified type."""
        if normalize_type == "none":
            return code
            
        if normalize_type == "docstring":
            code = self.remove_docstring(code)
            return code
        if normalize_type == "variables":
            code = self.remove_docstring(code)
            return self._replace_variables(code)
        elif normalize_type == "functions":
            code = self.remove_docstring(code)
            return self._replace_function_names(code)
        elif normalize_type == "both":
            code = self.remove_docstring(code)
            code = self._replace_function_names(code)
            return self._replace_variables(code)
        else:
            raise ValueError(f"Unknown normalization type: {normalize_type}")

    def _replace_variables(self, code: str) -> str:
        """Replace variable names with generic placeholders using AST-based normalization."""
        return self._normalize_variables_ast(code)

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
        example_printed = False
        
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
        
        print("Normalization type: ", normalize_type)
        print(doc_text)
        print()
        print()


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
    hf_api_key = os.getenv("HF_API_KEY")  #api key
    
    if not hf_api_key:
        raise ValueError("Please set HF_API_KEY environment variable")
        
    evaluator = CodeRetrievalEvaluator(
        hf_api_key=hf_api_key,
        model_name="meta-llama/Llama-3.1-70B-Instruct"
    )
    
    # Test different normalization strategies
    normalize_types = ["none", "docstring", "variables", "functions", "both"]
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
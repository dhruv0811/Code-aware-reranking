import re
import os 
import ast
import numpy as np
from typing import Dict, Set
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch

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



class ProgrammingSolutionsCorpus:
    def __init__(self,
                 embedding_model_name: str = "avsolatorio/GIST-large-Embedding-v0"
                 ):
        hf_api_key = os.getenv("HF_API_KEY")
        if hf_api_key is None:
            raise ValueError("Must provide HuggingFace API key")
    
        print("Initializing Embedding Model...")
        self.embedding_model_name = embedding_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )
        
        self.corpus = load_dataset("code-rag-bench/programming-solutions")
        self.vectorstore = None 
        self.normalize_type = None
        print("Corpus loaded!")
    

    def remove_docstring(self, code: str, task: str) -> str:
        """Remove docstring from Python code while preserving other comments."""
        humaneval_docstring_pattern = r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\''
        mbpp_docstring_pattern = r'^.*?(?=def\s)'

        if task == "humaneval":
            return re.sub(humaneval_docstring_pattern, '', code)
        elif task == "mbpp":
            return re.sub(mbpp_docstring_pattern, '', code, flags=re.DOTALL)
        
        raise ValueError(f"Unknown task: {task}")
    
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
    

    def normalize_code(self, code: str, normalize_type: str = "none", task: str = "humaneval") -> str:
        """Normalize code based on specified type."""
        if normalize_type == "none":
            return code
            
        # print("Normalizing code with type: ", normalize_type)

        code = self.remove_docstring(code, task=task)
        if normalize_type == "docstring":
            return code

        if normalize_type == "variables":
            return self._replace_variables(code)
        elif normalize_type == "functions":
            return self._replace_function_names(code)
        elif normalize_type == "both":
            code = self._replace_function_names(code)
            return self._replace_variables(code)
        else:
            raise ValueError(f"Unknown normalization type: {normalize_type}")


    def load(self, normalize_type: str = "none"):
        documents = []
        metadatas = []

        for item in self.corpus['train']:
            if item["meta"]["task_name"] == "humaneval":
                doc_text = item["text"]

                doc_text = self.normalize_code(doc_text, normalize_type, task="humaneval")
                
                idx = item["meta"]["task_id"]
                metadatas.append({
                    'source': f"programming-solutions_{idx}",
                    'index': idx,
                })
                documents.append(doc_text)
            else:
                doc_text = item["text"]

                doc_text = self.normalize_code(doc_text, normalize_type, task="mbpp")

                idx = item["meta"]["task_id"]
                metadatas.append({
                    'source': f"programming-solutions-mbpp_{idx}",
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

    def search(self, query: str, k: int = 5):
        results = self.vectorstore.similarity_search_with_score(
            query, 
            k
        )
        return results
    
    def compute_similarity(self, query: str, document: str) -> float:
        query_embedding = self.embedding_model.embed_query(query)
        doc_embedding = self.embedding_model.embed_documents([document])[0]
        
        query_embedding = np.array(query_embedding)
        doc_embedding = np.array(doc_embedding)
        
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        
        return similarity


if __name__ == "__main__":
    corpus = ProgrammingSolutionsCorpus()
    # corpus.load(normalize_type="both")
    corpus.load(normalize_type="both")
    print(corpus.search("Write a python function to remove first and last occurrence of a given character from the string", k=5))
    
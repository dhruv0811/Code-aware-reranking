import re
import os 
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class ProgrammingSolutionsCorpus:
    def __init__(self,
                 embedding_model_name: str = "avsolatorio/GIST-large-Embedding-v0"
                 ):
        hf_api_key = os.getenv("HF_API_KEY")
        if hf_api_key is None:
            raise ValueError("Must provide HuggingFace API key")
    
        print("Initializing Embedding Model...")
        self.embedding_model_name = embedding_model_name
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )
        
        self.corpus = load_dataset("code-rag-bench/programming-solutions")
        self.vectorstore = None 
        self.normalize_type = None
        print("Corpus loaded!")
    

    def remove_docstring(self, code: str) -> str:
        """Remove docstring from Python code while preserving other comments."""
        docstring_pattern = r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\''
        return re.sub(docstring_pattern, '', code)
    

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
    

    def normalize_code(self, code: str, normalize_type: str = "none") -> str:
        """Normalize code based on specified type."""
        if normalize_type == "none":
            return code
            
        print("Normalizing code with type: ", normalize_type)

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


    def load(self, normalize_type: str = "none"):
        documents = []
        metadatas = []

        for item in self.corpus['train']:
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

    def search(self, query: str, k: int = 5):
        results = self.vectorstore.similarity_search_with_score(
            query, 
            k
        )
        return results


if __name__ == "__main__":
    corpus = ProgrammingSolutionsCorpus()
    corpus.load(normalize_type="both")
    print(corpus.search("Test input!", k=5))
    
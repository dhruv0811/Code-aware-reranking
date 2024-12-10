import pickle
from pathlib import Path

def read_description_cache(cache_file_path):
    """
    Read and display examples from a description cache pickle file.
    """
    try:
        with open(cache_file_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        print(f"Total entries in cache: {len(cache_data)}\n")
        
        # Display first 5 entries
        print("Sample entries:")
        for i, (key, description) in enumerate(list(cache_data.items())[:15]):
            # Keys are in format: {code_hash}_{model_name}_{prompt_type}
            hash_part, model_name, prompt_type = key.split('_', 2)
            print(f"\nEntry {i + 1}:")
            print(f"Code Hash: {hash_part}")
            print(f"Model: {model_name}")
            print(f"Prompt Type: {prompt_type}")
            print(f"Description: {description}")
            print("-" * 80)

    except Exception as e:
        print(f"Error reading cache file: {e}")

# Usage
cache_path = "results/fixed_corpus_humaneval_reranker/cache_meta-llama_Llama-3.1-70B-Instruct_basic.pkl"
read_description_cache(cache_path)
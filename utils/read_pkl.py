import pickle
from pathlib import Path

def read_pickle_file(file_path):
    """Read and display contents of a pickle file"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        print(f"Successfully loaded pickle file: {file_path}")
        print("\nData type:", type(data))
        print("\nNumber of items:", len(data))
        
        # If it's a dictionary, show some sample keys and values
        if isinstance(data, dict):
            print("\nSample entries (first 5):")
            for i, (key, value) in enumerate(data.items()):
                if i >= 5:
                    break
                print(f"\nKey ({type(key)}): {key[:200]}..." if isinstance(key, str) else f"\nKey: {key}")
                print(f"Value ({type(value)}): {value[:500]}..." if isinstance(value, str) else f"Value: {value}")
                print("-----------------------------")
                print()
                print()
        
        return data
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error reading pickle file: {e}")

# Assuming the pickle file is in the pseudocode_cache directory
cache_dir = Path("pseudocode_cache")
pickle_file = cache_dir / "humaneval_pseudocode.pkl"

data = read_pickle_file(pickle_file)
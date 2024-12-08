from Corpus import ProgrammingSolutionsCorpus

def demonstrate_normalizations():
    # Initialize the corpus
    corpus = ProgrammingSolutionsCorpus()
    
    # Get the first entry's code for demonstration
    first_entry = corpus.corpus['train'][0]['text']
    
    # Demonstrate all normalization types
    print("Original Code:")
    print("-" * 80)
    print(first_entry)
    print("\n")
    
    normalization_types = ["docstring", "variables", "functions", "both"]
    
    for norm_type in normalization_types:
        print(f"Normalization Type: {norm_type}")
        print("-" * 80)
        normalized_code = corpus.normalize_code(first_entry, normalize_type=norm_type)
        print(normalized_code)
        print("\n")

if __name__ == "__main__":
    demonstrate_normalizations()
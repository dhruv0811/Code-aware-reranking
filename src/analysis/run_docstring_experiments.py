#!/usr/bin/env python3
"""
Run a direct comparison of normal vs docstring vs function normalization
to analyze the impact on code retrieval performance.
"""

import os
import sys
from pathlib import Path

# Add parent directory to the path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

try:
    from DocstringBiasExperiment import DocstringBiasComparison, run_normalization_comparison
except ImportError:
    print("Error importing DocstringBiasComparison. Trying alternative paths...")
    # Try with full path
    module_dir = Path(__file__).parent
    sys.path.append(str(module_dir))
    from DocstringBiasComparison import DocstringBiasComparison, run_normalization_comparison

def main():
    """Run the normalization comparison experiment."""
    # Check for API key
    if "HF_API_KEY" not in os.environ:
        print("Please set the HF_API_KEY environment variable")
        print("You can get an API key from https://huggingface.co/settings/tokens")
        return 1

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Compare normal vs docstring vs function normalization for retrieval")
    parser.add_argument("--dataset", type=str, default="openai_humaneval",
                        choices=["openai_humaneval", "code-rag-bench/mbpp"],
                        help="Dataset to use for queries")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of samples to process (None for all)")
    parser.add_argument("--output", type=str, default="/home/gganeshl/Code-aware-reranking/results/normalization_comparison",
                        help="Output directory for results")
    parser.add_argument("--models", nargs='+', default=["avsolatorio/GIST-large-Embedding-v0", "avsolatorio/GIST-Embedding-v0"],
                        help="Embedding models to test")
    
    args = parser.parse_args()
    
    # Run the comparison experiment
    print(f"Running normalization comparison on {args.dataset}")
    experiment = DocstringBiasComparison(
        embedding_model_names=args.models,
        output_dir=args.output,
        dataset_name=args.dataset
    )
    experiment.run_comparison(num_samples=args.samples)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
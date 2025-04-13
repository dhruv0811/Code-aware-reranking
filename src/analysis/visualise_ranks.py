import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Directory structure
BASE_DIR = Path("/home/gganeshl/Code-aware-reranking/results/normalization_comparison")
OUTPUT_DIR = BASE_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_results():
    """Load results from both HumanEval and MBPP directories"""
    results = {}
    
    # Load HumanEval results
    humaneval_files = glob.glob(str(BASE_DIR / "humaneval" / "*.json"))
    if humaneval_files:
        with open(humaneval_files[0], 'r') as f:
            results["humaneval"] = json.load(f)
            print(f"Loaded HumanEval results: {os.path.basename(humaneval_files[0])}")
    
    # Load MBPP results
    mbpp_files = glob.glob(str(BASE_DIR / "mbpp" / "*.json"))
    if mbpp_files:
        with open(mbpp_files[0], 'r') as f:
            results["mbpp"] = json.load(f)
            print(f"Loaded MBPP results: {os.path.basename(mbpp_files[0])}")
    
    return results

def extract_data_for_plots(results_data):
    """Extract hit rates and ranks from the results"""
    if not results_data:
        return None
    
    data = {}
    
    for model_result in results_data.get("model_results", []):
        model_name = model_result["embedding_model"].split("/")[-1]
        data[model_name] = {}
        
        # Get normalization results
        norm_results = model_result.get("normalization_results", {})
        
        # Extract hit rates and ranks
        for norm_type, metrics in norm_results.items():
            data[model_name][norm_type] = {
                "avg_rank": metrics.get("avg_rank", 0),
                "median_rank": metrics.get("median_rank", 0),
                "hit_rates": metrics.get("hit_rates", {})
            }
    
    return data

def create_rank_comparison_by_norm_type(dataset_name, data):
    """Create a line plot comparing ranks across normalization types"""
    if not data:
        print(f"No data available for {dataset_name}")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Define normalization types order for x-axis
    norm_types = ["none", "docstring", "functions"]
    
    # Define colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    
    # Plot lines for each model
    for i, (model_name, model_data) in enumerate(data.items()):
        # Get ranks for each normalization type
        ranks = []
        for norm_type in norm_types:
            if norm_type in model_data:
                ranks.append(model_data[norm_type]["avg_rank"])
            else:
                # Handle missing data
                ranks.append(None)
        
        # Plot line
        plt.plot(norm_types, ranks, 'o-', label=f"{model_name}", color=colors[i], linewidth=2, markersize=8)
        
        # Add value labels above each point
        for j, rank in enumerate(ranks):
            if rank is not None:
                plt.text(j, rank + 0.5, f"{rank:.2f}", ha='center', va='bottom', fontsize=9)
    
    plt.title(f"{dataset_name} Average Rank by Normalization Type", fontsize=14)
    plt.xlabel("Normalization Type", fontsize=12)
    plt.ylabel("Average Rank (lower is better)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set y-axis to start from 0 if possible
    current_ymin, current_ymax = plt.ylim()
    plt.ylim(0 if 0 < current_ymin else current_ymin, current_ymax * 1.1)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"{dataset_name.lower()}_rank_by_norm_type.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved rank comparison by normalization type to {output_path}")
    plt.close()

def create_hit_rate_comparison_by_norm_type(dataset_name, data, k_value=10):
    """Create a line plot comparing hit rates across normalization types for a specific k"""
    if not data:
        print(f"No data available for {dataset_name}")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Define normalization types order for x-axis
    norm_types = ["none", "docstring", "functions"]
    
    # Define colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    
    # Plot lines for each model
    for i, (model_name, model_data) in enumerate(data.items()):
        # Get hit rates for each normalization type
        hit_rates = []
        for norm_type in norm_types:
            if norm_type in model_data and str(k_value) in model_data[norm_type]["hit_rates"]:
                hit_rates.append(model_data[norm_type]["hit_rates"][str(k_value)])
            else:
                # Handle missing data
                hit_rates.append(None)
        
        # Plot line
        plt.plot(norm_types, hit_rates, 'o-', label=f"{model_name}", color=colors[i], linewidth=2, markersize=8)
        
        # Add value labels above each point
        for j, rate in enumerate(hit_rates):
            if rate is not None:
                plt.text(j, rate + 0.02, f"{rate:.3f}", ha='center', va='bottom', fontsize=9)
    
    plt.title(f"{dataset_name} Hit Rate @{k_value} by Normalization Type", fontsize=14)
    plt.xlabel("Normalization Type", fontsize=12)
    plt.ylabel(f"Hit Rate @{k_value} (higher is better)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set y-axis to be between 0 and 1
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"{dataset_name.lower()}_hitrate_k{k_value}_by_norm_type.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved hit rate comparison by normalization type to {output_path}")
    plt.close()

def create_combined_comparison(results):
    """Create a figure comparing both datasets side by side for each normalization method"""
    # Extract data for both datasets
    humaneval_data = extract_data_for_plots(results.get("humaneval"))
    mbpp_data = extract_data_for_plots(results.get("mbpp"))
    
    if not humaneval_data or not mbpp_data:
        print("Not enough data for combined comparison")
        return
    
    # Define normalization types
    norm_types = ["none", "docstring", "functions"]
    
    # Create a plot for each model 
    for model_name in humaneval_data.keys():
        if model_name not in mbpp_data:
            continue
            
        plt.figure(figsize=(12, 6))
        
        # Get data for this model
        humaneval_model_data = humaneval_data[model_name]
        mbpp_model_data = mbpp_data[model_name]
        
        # Plot HumanEval ranks
        humaneval_ranks = [humaneval_model_data.get(norm, {}).get("avg_rank", None) for norm in norm_types]
        plt.plot(norm_types, humaneval_ranks, 'o-', label="HumanEval", color='blue', linewidth=2, markersize=8)
        
        # Plot MBPP ranks
        mbpp_ranks = [mbpp_model_data.get(norm, {}).get("avg_rank", None) for norm in norm_types]
        plt.plot(norm_types, mbpp_ranks, 'o-', label="MBPP", color='red', linewidth=2, markersize=8)
        
        # Add value labels
        for i, rank in enumerate(humaneval_ranks):
            if rank is not None:
                plt.text(i, rank + 0.5, f"{rank:.2f}", ha='center', va='bottom', fontsize=9, color='blue')
                
        for i, rank in enumerate(mbpp_ranks):
            if rank is not None:
                plt.text(i, rank + 0.5, f"{rank:.2f}", ha='center', va='bottom', fontsize=9, color='red')
        
        plt.title(f"Average Rank Comparison - {model_name}", fontsize=14)
        plt.xlabel("Normalization Type", fontsize=12)
        plt.ylabel("Average Rank (lower is better)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Set y-axis to start from 0 if possible
        current_ymin, current_ymax = plt.ylim()
        plt.ylim(0 if 0 < current_ymin else current_ymin, current_ymax * 1.1)
        
        plt.tight_layout()
        
        output_path = OUTPUT_DIR / f"combined_{model_name}_rank_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined rank comparison for {model_name} to {output_path}")
        plt.close()

def create_normalization_improvement_plot(dataset_name, data):
    """Create a bar chart showing improvement from baseline for each normalization type"""
    if not data:
        print(f"No data available for {dataset_name}")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Collect data for plotting
    models = []
    docstring_improvements = []
    functions_improvements = []
    
    for model_name, model_data in data.items():
        # Skip if we don't have all normalization types
        if 'none' not in model_data or 'docstring' not in model_data or 'functions' not in model_data:
            continue
        
        # Calculate percentage improvement from baseline
        baseline_rank = model_data['none']['avg_rank']
        docstring_rank = model_data['docstring']['avg_rank']
        functions_rank = model_data['functions']['avg_rank']
        
        # Calculate percentage improvement (positive means better/lower rank)
        docstring_improvement = (baseline_rank - docstring_rank) / baseline_rank * 100
        functions_improvement = (baseline_rank - functions_rank) / baseline_rank * 100
        
        models.append(model_name)
        docstring_improvements.append(docstring_improvement)
        functions_improvements.append(functions_improvement)
    
    # Create grouped bar chart
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, docstring_improvements, width, label='Docstring Normalization', color='red', alpha=0.7)
    plt.bar(x + width/2, functions_improvements, width, label='Function Normalization', color='green', alpha=0.7)
    
    # Add reference line at 0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add percentage labels
    for i, v in enumerate(docstring_improvements):
        plt.text(i - width/2, v + (1 if v >= 0 else -3), f"{v:.1f}%", ha='center', fontsize=9)
        
    for i, v in enumerate(functions_improvements):
        plt.text(i + width/2, v + (1 if v >= 0 else -3), f"{v:.1f}%", ha='center', fontsize=9)
    
    plt.title(f"{dataset_name} Percentage Improvement in Rank vs. Baseline", fontsize=14)
    plt.xlabel("Embedding Model", fontsize=12)
    plt.ylabel("% Improvement (positive is better)", fontsize=12)
    plt.xticks(x, models)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"{dataset_name.lower()}_percent_improvement.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved percentage improvement plot to {output_path}")
    plt.close()

def main():
    """Main function to create all visualizations"""
    # Load results
    results = load_results()
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Process HumanEval results
    if "humaneval" in results:
        humaneval_data = extract_data_for_plots(results["humaneval"])
        create_rank_comparison_by_norm_type("HumanEval", humaneval_data)
        create_hit_rate_comparison_by_norm_type("HumanEval", humaneval_data, k_value=10)
        create_hit_rate_comparison_by_norm_type("HumanEval", humaneval_data, k_value=1) 
        create_normalization_improvement_plot("HumanEval", humaneval_data)
    
    # Process MBPP results
    if "mbpp" in results:
        mbpp_data = extract_data_for_plots(results["mbpp"])
        create_rank_comparison_by_norm_type("MBPP", mbpp_data)
        create_hit_rate_comparison_by_norm_type("MBPP", mbpp_data, k_value=10)
        create_hit_rate_comparison_by_norm_type("MBPP", mbpp_data, k_value=1)
        create_normalization_improvement_plot("MBPP", mbpp_data)
    
    # Create combined comparison
    if "humaneval" in results and "mbpp" in results:
        create_combined_comparison(results)
    
    print("All visualizations completed!")

if __name__ == "__main__":
    main()
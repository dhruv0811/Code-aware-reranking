import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import cm
import matplotlib.patches as mpatches

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

def create_comprehensive_rank_plots(results):
    """
    Create separate rank plots for each dataset showing average ranks 
    across different normalization types
    """
    norm_types = ["none", "docstring", "functions"]
    
    # Define distinct marker styles
    marker_styles = ['o', 's', 'D', '^', 'v']
    
    # Define strongly contrasting color palettes for each dataset
    humaneval_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue, Orange, Green, Red, Purple
    mbpp_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']  # Red, Blue, Green, Purple, Orange
    
    humaneval_data = extract_data_for_plots(results.get("humaneval"))
    mbpp_data = extract_data_for_plots(results.get("mbpp"))
    
    # Create a separate plot for each dataset
    datasets = [("HumanEval", humaneval_data, humaneval_colors), 
                ("MBPP", mbpp_data, mbpp_colors)]
    
    for dataset_name, data, color_palette in datasets:
        if not data:
            print(f"No {dataset_name} data available")
            continue
            
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        min_rank, max_rank = float('inf'), 0
        
        for i, (model_name, model_data) in enumerate(data.items()):
            ranks = []
            for norm_type in norm_types:
                if norm_type in model_data:
                    rank = model_data[norm_type]["avg_rank"]
                    ranks.append(rank)
                    min_rank = min(min_rank, rank)
                    max_rank = max(max_rank, rank)
                else:
                    ranks.append(None)
            
            # Use distinct colors and markers for each model
            ax.plot(norm_types, ranks, marker=marker_styles[i % len(marker_styles)], 
                   linestyle='-', label=f"{model_name}", 
                   color=color_palette[i % len(color_palette)], 
                   linewidth=2, markersize=10)
            
            # Add text labels for rank values
            for j, rank in enumerate(ranks):
                if rank is not None:
                    ax.text(j, rank + 0.5, f"{rank:.1f}", ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f"{dataset_name}: Average Ranks Across Normalization Types", fontsize=16)
        ax.set_xlabel("Normalization Type", fontsize=12)
        ax.set_ylabel("Average Rank (lower is better)", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Improve the legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(3, len(data)), 
                 fontsize=10, frameon=True, facecolor='white', edgecolor='lightgray')
        
        plt.tight_layout()
        output_path = OUTPUT_DIR / f"{dataset_name.lower()}_rank_summary.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {dataset_name} rank summary plot to {output_path}")
        plt.close()

def create_improvement_plots(results):
    """
    Create separate improvement plots for HumanEval and MBPP, showing percentage improvements
    for each model and normalization type
    """
    humaneval_data = extract_data_for_plots(results.get("humaneval"))
    mbpp_data = extract_data_for_plots(results.get("mbpp"))
    
    norm_types = ["docstring", "functions"]
    datasets = [
        ("HumanEval", humaneval_data, cm.Greens(np.linspace(0.4, 0.8, 5))), 
        ("MBPP", mbpp_data, cm.RdPu(np.linspace(0.4, 0.8, 5)))
    ]
    
    # Create a separate plot for each dataset
    for dataset_name, data, color_palette in datasets:
        if not data:
            print(f"No {dataset_name} data available for improvement plot")
            continue
        
        all_models = sorted(data.keys())
        num_models = len(all_models)
        
        if num_models == 0:
            print(f"No models found for {dataset_name}")
            continue
        
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        
        # Calculate positions for grouped bars
        bar_width = 0.35
        index = np.arange(num_models)
        
        docstring_improvements = []
        functions_improvements = []
        
        for model_name in all_models:
            if 'none' not in data[model_name]:
                docstring_improvements.append(0)
                functions_improvements.append(0)
                continue
                
            baseline_rank = data[model_name]['none']['avg_rank']
            
            # Calculate improvement for docstring normalization
            if 'docstring' in data[model_name]:
                norm_rank = data[model_name]['docstring']['avg_rank']
                improvement = (baseline_rank - norm_rank) / baseline_rank * 100
                docstring_improvements.append(improvement)
            else:
                docstring_improvements.append(0)
                
            # Calculate improvement for functions normalization
            if 'functions' in data[model_name]:
                norm_rank = data[model_name]['functions']['avg_rank']
                improvement = (baseline_rank - norm_rank) / baseline_rank * 100
                functions_improvements.append(improvement)
            else:
                functions_improvements.append(0)
        
        # Plot bars with hatching for better distinction
        docstring_bars = ax.bar(index - bar_width/2, docstring_improvements, bar_width,
                color='#ff7f0e', hatch='\\\\', label='Docstring')
        
        functions_bars = ax.bar(index + bar_width/2, functions_improvements, bar_width,
                 color='#2ca02c', hatch='//', label='Functions')
        
        # Add value labels
        for i, v in enumerate(docstring_improvements):
            ax.text(i - bar_width/2, v + (1 if v >= 0 else -3),
                   f"{v:.1f}%", ha='center', fontsize=9, color='black')
                   
        for i, v in enumerate(functions_improvements):
            ax.text(i + bar_width/2, v + (1 if v >= 0 else -3),
                   f"{v:.1f}%", ha='center', fontsize=9, color='black')
        
        # Add a horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Set labels and title
        ax.set_title(f"{dataset_name}: Percentage Improvement vs. Baseline", fontsize=16)
        ax.set_ylabel("% Improvement (positive is better)", fontsize=12)
        ax.set_xticks(index)
        ax.set_xticklabels([model_name.split("-")[-1] for model_name in all_models], rotation=30, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add legend
        ax.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        output_path = OUTPUT_DIR / f"{dataset_name.lower()}_improvement_summary.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {dataset_name} improvement summary plot to {output_path}")
        plt.close()

def create_hit_rate_line_plots(results):
    """
    Create separate line plots for each dataset showing hit rates across k values
    """
    humaneval_data = extract_data_for_plots(results.get("humaneval"))
    mbpp_data = extract_data_for_plots(results.get("mbpp"))
    
    norm_types = ["none", "docstring", "functions"]
    
    # Get all k values
    k_values = set()
    for data in [humaneval_data, mbpp_data]:
        if data:
            for model_data in data.values():
                for norm_data in model_data.values():
                    k_values.update(map(int, norm_data["hit_rates"].keys()))
    
    k_values = sorted(k_values)
    
    if not k_values:
        print("No k values found in the data")
        return
    
    # Create line plots for each dataset
    datasets = [
        ("HumanEval", humaneval_data, ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']), 
        ("MBPP", mbpp_data, ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00'])
    ]
    
    # Define distinct line styles and colors for normalization types
    norm_styles = {
        "none": "-",
        "docstring": "--",
        "functions": "-."
    }
    
    # Fixed colors for normalization types (regardless of dataset)
    norm_colors = {
        "none": "#1f77b4",     # Blue
        "docstring": "#ff7f0e", # Orange
        "functions": "#2ca02c"  # Green
    }
    
    # Define distinct markers for models
    markers = ['o', 's', 'D', '^', 'v']
    
    for dataset_name, data, color_palette in datasets:
        if not data:
            print(f"No {dataset_name} data available for hit rate plot")
            continue
        
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        
        # Plot each model and normalization type with distinct visual properties
        legend_handles = []
        for model_idx, (model_name, model_data) in enumerate(data.items()):
            model_color = color_palette[model_idx % len(color_palette)]
            model_marker = markers[model_idx % len(markers)]
            
            for norm_type in norm_types:
                if norm_type not in model_data:
                    continue
                
                hit_rate_data = model_data[norm_type]["hit_rates"]
                
                if not hit_rate_data:
                    continue
                
                x_vals = []
                y_vals = []
                
                for k in k_values:
                    if str(k) in hit_rate_data:
                        x_vals.append(k)
                        y_vals.append(hit_rate_data[str(k)])
                
                if not x_vals:
                    continue
                
                line = ax.plot(x_vals, y_vals, 
                         label=f"{model_name} - {norm_type}",
                         linestyle=norm_styles[norm_type],
                         marker=model_marker,
                         markersize=8,
                         linewidth=2,
                         color=norm_colors[norm_type])
        
        ax.set_title(f"{dataset_name}: Hit Rates vs. k Values", fontsize=16)
        ax.set_xlabel("k value", fontsize=12)
        ax.set_ylabel("Hit Rate (R@k)", fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        # Create a custom legend with models and normalization types separately
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(3, len(data)*len(norm_types)), 
                 fontsize=10)
        
        plt.tight_layout()
        output_path = OUTPUT_DIR / f"{dataset_name.lower()}_hit_rate_lines.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {dataset_name} hit rate line plot to {output_path}")
        plt.close()

def create_hit_rate_bar_plots(results):
    """
    Create bar plots showing hit rates at a specific k value for each dataset
    """
    humaneval_data = extract_data_for_plots(results.get("humaneval"))
    mbpp_data = extract_data_for_plots(results.get("mbpp"))
    
    norm_types = ["none", "docstring", "functions"]
    
    # Get all k values
    k_values = set()
    for data in [humaneval_data, mbpp_data]:
        if data:
            for model_data in data.values():
                for norm_data in model_data.values():
                    k_values.update(map(int, norm_data["hit_rates"].keys()))
    
    k_values = sorted(k_values)
    
    if not k_values:
        print("No k values found in the data")
        return
    
    # Select a k value for bar plots
    if len(k_values) == 1:
        selected_k = k_values[0]
    else:
        k_options = [k for k in k_values if k <= 10]
        selected_k = max(k_options) if k_options else min(k_values)
    
    # Create bar plots for each dataset
    datasets = [
        ("HumanEval", humaneval_data, ['#1f77b4', '#ff7f0e', '#2ca02c']), 
        ("MBPP", mbpp_data, ['#e41a1c', '#377eb8', '#4daf4a'])
    ]
    
    for dataset_name, data, color_palette in datasets:
        if not data:
            print(f"No {dataset_name} data available for hit rate bar plot")
            continue
        
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        
        models = list(data.keys())
        num_models = len(models)
        
        # Set the positions and width of the bars
        bar_width = 0.25
        positions = np.arange(num_models)
        
        # Create bars for each normalization type
        for i, norm_type in enumerate(norm_types):
            hit_rates = []
            
            for model_name in models:
                if (norm_type in data[model_name] and 
                    str(selected_k) in data[model_name][norm_type]["hit_rates"]):
                    hit_rates.append(data[model_name][norm_type]["hit_rates"][str(selected_k)])
                else:
                    hit_rates.append(0)
            
            offset = (i - 1) * bar_width
            # Use consistent colors for normalization types
            norm_color = {
                'none': '#1f77b4',     # Blue
                'docstring': '#ff7f0e', # Orange
                'functions': '#2ca02c'  # Green
            }[norm_type]
            
            bars = ax.bar(positions + offset, hit_rates, bar_width, 
                     label=norm_type, color=norm_color, 
                     hatch=['', '\\\\', '//'][i])
            
            # Add value labels
            for j, v in enumerate(hit_rates):
                if v > 0:
                    ax.text(positions[j] + offset, v + 0.02, f"{v:.2f}", 
                           ha='center', fontsize=8, color='black')
        
        # Set chart properties
        ax.set_title(f"{dataset_name}: Hit Rate @{selected_k} by Model and Normalization", fontsize=16)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel(f"Hit Rate (R@{selected_k})", fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(positions)
        ax.set_xticklabels([model_name.split("-")[-1] for model_name in models], rotation=30, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
        
        plt.tight_layout()
        output_path = OUTPUT_DIR / f"{dataset_name.lower()}_hit_rate_bars_k{selected_k}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {dataset_name} hit rate bar plot to {output_path}")
        plt.close()

def main():
    """Main function to create all visualizations"""
    # Load results
    results = load_results()
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Create separate visualizations
    create_comprehensive_rank_plots(results)
    create_improvement_plots(results)
    create_hit_rate_line_plots(results)
    create_hit_rate_bar_plots(results)
    
    print("All visualizations completed!")

if __name__ == "__main__":
    main()
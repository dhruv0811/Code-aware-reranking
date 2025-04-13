import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_recall_scores(csv_file, include_reranking=False, output_file=None):
    """
    Create a simple line plot showing raw recall scores across normalization levels.
    
    Args:
        csv_file: Path to the CSV file with retrieval results
        include_reranking: Whether to include reranking results on the plot
        output_file: Path to save the output plot (default is "recall_plot.png")
    """
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Create a friendly name mapping for normalization types
    norm_mapping = {
        'none': 'No Normalization',
        'docstring': 'Docstring Removal',
        'variables': 'Variable Renaming',
        'functions': 'Function Renaming',
        'both': 'Full Normalization'
    }
    
    # Define normalization level for proper ordering
    norm_order = ['none', 'docstring', 'variables', 'functions', 'both']
    
    # Filter rows where rerank_k=5 for consistency
    if 'rerank_k' in df.columns:
        df = df[df['rerank_k'] == 5]
    
    # Create new dataframe for plotting
    plot_data = []
    
    for norm_type in norm_order:
        row = df[df['normalization_type'] == norm_type]
        if not row.empty:
            row = row.iloc[0]
            
            # Add baseline metrics
            baseline_col = 'baseline_recall@1'
            if baseline_col in df.columns:
                value = row[baseline_col]
                plot_data.append({
                    'normalization': norm_mapping[norm_type],
                    'metric': 'Baseline Recall@1',
                    'value': value,
                    'norm_order': norm_order.index(norm_type)
                })
            
            # Add reranked metrics if requested
            if include_reranking:
                reranked_col = 'reranked_recall@1'
                if reranked_col in df.columns:
                    value = row[reranked_col]
                    plot_data.append({
                        'normalization': norm_mapping[norm_type],
                        'metric': 'Reranked Recall@1',
                        'value': value,
                        'norm_order': norm_order.index(norm_type)
                    })
    
    # Convert to dataframe
    plot_df = pd.DataFrame(plot_data)
    
    # Sort by normalization order
    plot_df = plot_df.sort_values('norm_order')
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Use line plot for the visualization
    sns.lineplot(
        data=plot_df,
        x='normalization',
        y='value',
        hue='metric',
        marker='o',
        markersize=10,
        linewidth=2
    )
    
    # Customize the plot
    plt.title('Retrieval Performance Across Normalization Levels (Humaneval Dataset)', fontsize=16)
    plt.xlabel('Normalization Type', fontsize=14)
    plt.ylabel('Recall@1', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Metric', fontsize=12)
    
    # Set y-axis to start from 0 and end at 1
    plt.ylim(0, 1.05)
    
    # Add annotations for the values
    for i, row in plot_df.iterrows():
        plt.annotate(
            f"{row['value']:.2f}",
            (row['normalization'], row['value']),
            xytext=(0, 7),
            textcoords='offset points',
            ha='center',
            fontsize=9
        )
    
    plt.tight_layout()
    
    # Save the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        output_name = "recall_plot.png"
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_name}")

def main():
    parser = argparse.ArgumentParser(description='Create a simple line plot of recall scores')
    parser.add_argument('csv_file', help='CSV file with retrieval results')
    parser.add_argument('--include-reranking', action='store_true', 
                        help='Include reranked results on the plot')
    parser.add_argument('--output', help='Output file for the plot (default: recall_plot.png)')
    
    args = parser.parse_args()
    
    # Create the plot
    plot_recall_scores(args.csv_file, args.include_reranking, args.output)

if __name__ == "__main__":
    main()
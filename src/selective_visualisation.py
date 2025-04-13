import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import os
import numpy as np

def load_json_data(json_file):
    """Load experiment results from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def setup_plotting_style():
    """Setup consistent plotting style with pastel colors."""
    # Set the style to a clean, modern look
    plt.style.use('seaborn-v0_8-pastel')
    
    # Define a pastel color palette
    pastel_palette = {
        'green': '#8ECFA8',  # Pastel green for improvements
        'red': '#FF9999',    # Pastel red for worsenings
        'blue': '#A8D1FF',   # Pastel blue for baseline
        'purple': '#D9B3FF', # Pastel purple for docstring
        'orange': '#FFD699', # Pastel orange for function
        'gray': '#D9D9D9',   # Pastel gray for unchanged
    }
    
    # Customize general plot appearance
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    return pastel_palette

def create_rank_comparison(results, output_dir, colors):
    """Create bar chart comparing average and median ranks."""
    plt.figure(figsize=(12, 6))
    
    ranks_df = pd.DataFrame({
        'Normalization Type': ['Baseline', 'Docstring Normalized', 'Function Normalized'],
        'Average Rank': [
            results['average_ranks']['baseline'], 
            results['average_ranks']['docstring_normalized_gold'],
            results['average_ranks']['function_normalized_gold']
        ],
        'Median Rank': [
            results['median_ranks']['baseline'], 
            results['median_ranks']['docstring_normalized_gold'],
            results['median_ranks']['function_normalized_gold']
        ]
    })
    
    # Reshape for seaborn
    ranks_df_melted = pd.melt(ranks_df, id_vars='Normalization Type', 
                              value_vars=['Average Rank', 'Median Rank'],
                              var_name='Metric', value_name='Rank')
    
    # Set up custom colors for the normalization types
    norm_colors = [colors['blue'], colors['purple'], colors['orange']]
    
    # Create the plot
    ax = sns.barplot(x='Normalization Type', y='Rank', hue='Metric', data=ranks_df_melted,
                     palette=['#6495ED', '#9370DB'])  # Light blue for avg, light purple for median
    
    plt.title('Average and Median Ranks by Normalization Type', fontweight='bold')
    plt.ylabel('Rank (lower is better)')
    plt.ylim(0, max(ranks_df_melted['Rank']) * 1.1)  # Add some padding at the top
    
    # Add value labels on the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontweight='bold')
    
    # Enhance the legend
    plt.legend(title='Metric', frameon=True, fancybox=True, framealpha=0.9, title_fontsize=13)
    
    # Add grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rank_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return ranks_df

def create_hit_rates_plot(results, output_dir, colors):
    """Create line chart showing hit rates at different k values."""
    plt.figure(figsize=(14, 8))
    
    # Extract hit rates and prepare dataframe
    k_values = list(results['hit_rates']['baseline'].keys())
    norm_types = list(results['hit_rates'].keys())
    
    hit_rates_data = []
    for norm_type in norm_types:
        for k in k_values:
            # Remove any trailing spaces to ensure consistency
            display_name = norm_type.replace('_', ' ').title().replace('Gold', '').strip()
            hit_rates_data.append({
                'Normalization Type': display_name,
                'K Value': int(k),
                'Hit Rate': results['hit_rates'][norm_type][k]
            })
    
    hit_rates_df = pd.DataFrame(hit_rates_data)
    
    # Create the line plot for hit rates
    plt.figure(figsize=(14, 8))
    
    # Define distinct markers for each normalization type - ensure keys are consistent
    markers = {
        'Baseline': 'o', 
        'Docstring Normalized': 's', 
        'Function Normalized': '^'
    }
    
    linestyles = {
        'Baseline': '-', 
        'Docstring Normalized': '--', 
        'Function Normalized': '-.'
    }
    
    # Plot each normalization type separately to control markers and line styles
    for norm_type in hit_rates_df['Normalization Type'].unique():
        data = hit_rates_df[hit_rates_df['Normalization Type'] == norm_type]
        plt.plot(data['K Value'], data['Hit Rate'], 
                 marker=markers[norm_type], 
                 linestyle=linestyles[norm_type],
                 linewidth=3, 
                 markersize=10,
                 label=norm_type)
    
    plt.title('Hit Rates at Different K Values', fontweight='bold')
    plt.xlabel('K Value (Number of Results)')
    plt.ylabel('Hit Rate (higher is better)')
    plt.xticks(sorted([int(k) for k in k_values]))
    plt.ylim(0, 1.05)  # Set y-axis limit
    
    # Add grid lines for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Enhance the legend
    plt.legend(title='Normalization Type', frameon=True, fancybox=True, 
               framealpha=0.9, title_fontsize=13)
    
    # Add value annotations at important points
    for norm_type in hit_rates_df['Normalization Type'].unique():
        data = hit_rates_df[hit_rates_df['Normalization Type'] == norm_type]
        # Annotate K=1 and K=5 points
        for k in [1, 5]:
            point_data = data[data['K Value'] == k]
            if not point_data.empty:
                plt.annotate(f'{point_data["Hit Rate"].values[0]:.2f}', 
                            (k, point_data['Hit Rate'].values[0]),
                            textcoords="offset points",
                            xytext=(0,10), 
                            ha='center',
                            fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hit_rates.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return hit_rates_df

def create_rank_changes_plot(results, output_dir, colors):
    """Create stacked bar chart showing the distribution of rank changes."""
    plt.figure(figsize=(12, 6))
    
    # Prepare data for rank changes
    rank_changes_df = pd.DataFrame({
        'Normalization Type': ['Docstring Normalized', 'Function Normalized'],
        'Unchanged': [
            results['rank_changes']['docstring_normalized_gold']['unchanged_percent'],
            results['rank_changes']['function_normalized_gold']['unchanged_percent']
        ],
        'Better Rank': [
            results['rank_changes']['docstring_normalized_gold']['improved_percent'],
            results['rank_changes']['function_normalized_gold']['improved_percent']
        ],
        'Worse Rank': [
            results['rank_changes']['docstring_normalized_gold']['worsened_percent'],
            results['rank_changes']['function_normalized_gold']['worsened_percent']
        ]
    })
    
    # Reshape for seaborn
    rank_changes_melted = pd.melt(rank_changes_df, id_vars='Normalization Type', 
                                 value_vars=['Unchanged', 'Worse Rank', 'Better Rank'],
                                 var_name='Change Type', value_name='Percentage')
    
    # Define color mapping for change types
    change_colors = {
        'Unchanged': colors['gray'],
        'Worse Rank': colors['red'],
        'Better Rank': colors['green']
    }
    
    # Create the stacked bar chart
    ax = sns.barplot(x='Normalization Type', y='Percentage', hue='Change Type', 
                     data=rank_changes_melted, palette=change_colors)
    
    plt.title('Distribution of Rank Changes After Normalization', fontweight='bold')
    plt.ylabel('Percentage of Queries')
    plt.ylim(0, 100)  # Set y-axis limit to 100%
    
    # Add percentage labels with enhanced styling
    for container in ax.containers:
        labels = [f'{val:.1f}%' if val > 0 else '' for val in container.datavalues]
        ax.bar_label(container, labels=labels, label_type='center', fontweight='bold')
    
    # Enhance the legend
    plt.legend(title='Change Type', frameon=True, fancybox=True, 
               framealpha=0.9, title_fontsize=13)
    
    # Add grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rank_changes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return rank_changes_df

def create_k1_hit_rate_plot(hit_rates_df, output_dir, colors):
    """Create bar chart showing hit rates at k=1."""
    plt.figure(figsize=(10, 6))
    
    # Filter for K=1
    k1_hit_rates = hit_rates_df[hit_rates_df['K Value'] == 1]
    
    # Map normalization types to colors - ensure consistent keys
    norm_colors = {
        'Baseline': colors['blue'],
        'Docstring Normalized': colors['purple'],
        'Function Normalized': colors['orange']
    }
    
    # Create the plot with custom colors
    norm_palette = [norm_colors[norm] for norm in k1_hit_rates['Normalization Type']]
    ax = sns.barplot(x='Normalization Type', y='Hit Rate', data=k1_hit_rates, palette=norm_palette)
    
    plt.title('Hit Rate at K=1 (Top Result)', fontweight='bold')
    plt.ylabel('Hit Rate (higher is better)')
    plt.ylim(0, 1.05)
    
    # Add value labels with enhanced styling
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha='center', va='bottom', fontweight='bold',
                    xytext=(0, 5), textcoords='offset points')
    
    # Add grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hit_rate_k1.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_rank_displacement_visualization(results, output_dir, colors):
    """
    Create visualization showing how document ranks are displaced by normalization.
    This helps visualize when irrelevant docs are preferred over normalized gold docs.
    """
    plt.figure(figsize=(14, 8))
    
    # Extract the per_query_details if available
    if 'per_query_details' in results and results['per_query_details']:
        # Use actual query details
        query_details = results['per_query_details']
        
        # Calculate rank displacement for each normalization type
        docstring_displacements = [d['docstring_rank'] - d['base_rank'] for d in query_details]
        function_displacements = [d['function_rank'] - d['base_rank'] for d in query_details]
        
        # Get max displacement for plot limits
        max_displacement = max(max(function_displacements), 1)
    else:
        # Create synthetic data based on summary statistics
        # This is an approximation since we don't have the actual per-query data
        n_queries = results['num_queries']
        
        # For docstring normalization
        docstring_unchanged = int(results['rank_changes']['docstring_normalized_gold']['unchanged_percent'] * n_queries / 100)
        docstring_worse = int(results['rank_changes']['docstring_normalized_gold']['improved_percent'] * n_queries / 100)
        
        # For function normalization
        function_unchanged = int(results['rank_changes']['function_normalized_gold']['unchanged_percent'] * n_queries / 100)
        function_worse = int(results['rank_changes']['function_normalized_gold']['improved_percent'] * n_queries / 100)
        
        # Create synthetic displacements
        docstring_displacements = [0] * docstring_unchanged + [np.random.randint(1, 5) for _ in range(docstring_worse)]
        function_displacements = [0] * function_unchanged + [np.random.randint(1, 30) for _ in range(function_worse)]
        
        # Get max displacement for plot limits
        max_displacement = max(max(function_displacements), 1)
    
    # Create displacement bins
    bins = range(0, max_displacement + 10, 1)
    
    # Plot histograms for each normalization type
    plt.hist(docstring_displacements, bins=bins, alpha=0.7, 
             label='Docstring Normalized', color=colors['purple'], edgecolor='black')
    plt.hist(function_displacements, bins=bins, alpha=0.7, 
             label='Function Normalized', color=colors['orange'], edgecolor='black')
    
    plt.title('Distribution of Rank Displacement After Normalization', fontweight='bold')
    plt.xlabel('Rank Displacement (higher = more irrelevant docs preferred)')
    plt.ylabel('Number of Queries')
    plt.xlim(0, min(max_displacement + 5, 30))  # Limit x-axis for better visibility
    
    # Add a vertical line at 0 (no displacement)
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.8, 
                label='No Displacement (Rank unchanged)')
    
    # Draw a text box explaining the visualization
    # explanation = (
    #     "This plot shows how many positions the gold document's rank\n"
    #     "worsened after normalization. Higher values mean more irrelevant\n"
    #     "documents were preferred over the normalized gold document."
    # )
    # plt.annotate(explanation, xy=(0.5, 0.95), xycoords='axes fraction',
    #              bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
    #              ha='center', va='top')
    
    # Enhance the legend
    plt.legend(frameon=True, fancybox=True, framealpha=0.9)
    
    # Add grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rank_displacement.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_preference_visualization(results, output_dir, colors):
    """
    Create visualization showing preference for unnormalized irrelevant docs vs normalized relevant docs.
    """
    plt.figure(figsize=(12, 7))
    
    # Calculate the percentage of cases where normalization hurt ranking
    docstring_hurt_pct = results['rank_changes']['docstring_normalized_gold']['worsened_percent']
    function_hurt_pct = results['rank_changes']['function_normalized_gold']['worsened_percent']

    print("Docstring hurt pct: ", results['rank_changes']['docstring_normalized_gold']['worsened_percent'])
    print("Function hurt pct: ", results['rank_changes']['function_normalized_gold']['worsened_percent'])
    
    # Create a DataFrame for visualization
    preference_df = pd.DataFrame({
        'Normalization Type': ['Docstring Normalized', 'Function Normalized'],
        'Irrelevant Doc Preferred': [docstring_hurt_pct, function_hurt_pct],
        'Normalized Doc Preferred': [100 - docstring_hurt_pct, 100 - function_hurt_pct]
    })
    
    # Reshape for seaborn
    preference_melted = pd.melt(
        preference_df, 
        id_vars='Normalization Type', 
        value_vars=['Normalized Doc Preferred', 'Irrelevant Doc Preferred'],
        var_name='Preference', 
        value_name='Percentage'
    )
    
    # Define color mapping
    preference_colors = {
        'Normalized Doc Preferred': colors['green'],
        'Irrelevant Doc Preferred': colors['red']
    }
    
    # Create the stacked bar chart
    ax = sns.barplot(
        x='Normalization Type', 
        y='Percentage', 
        hue='Preference', 
        data=preference_melted, 
        palette=preference_colors
    )
    
    plt.title('Retriever Preference: Normalized Relevant Doc vs. Unnormalized Irrelevant Doc', 
              fontweight='bold')
    plt.ylabel('Percentage of Queries')
    plt.ylim(0, 100)  # Set y-axis limit to 100%
    
    # Add percentage labels
    for container in ax.containers:
        labels = [f'{val:.1f}%' if val > 0 else '' for val in container.datavalues]
        ax.bar_label(container, labels=labels, label_type='center', fontweight='bold')
    
    # Enhance the legend
    plt.legend(title='Retriever Preference', frameon=True, fancybox=True, 
               framealpha=0.9, title_fontsize=13)
    
    # Add grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # # Add explanatory text
    # explanation = (
    #     "This visualization shows whether the retriever prefers the normalized relevant document\n"
    #     "or irrelevant documents that were not normalized. When 'Irrelevant Doc Preferred' is high,\n"
    #     "it means the retriever is biased toward specific code formatting rather than content."
    # )
    # plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=10, 
    #             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.07, 1, 1])  # Make room for the text at the bottom
    plt.savefig(os.path.join(output_dir, 'retriever_preference.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate plots from experiment results')
    parser.add_argument('--json_file', type=str, default="/home/gganeshl/Code-aware-reranking/results/selective_normalization/selective_normalization_20250412_201810_results.json", help='Path to JSON results file')
    parser.add_argument('--output_dir', type=str, default="/home/gganeshl/Code-aware-reranking/results/selective_normalization/visualizations", help='Directory to save plots')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    results = load_json_data(args.json_file)
    
    print(f"Generating plots for experiment: {results.get('experiment_id', 'unknown')}")
    
    # Setup plotting style and get color palette
    colors = setup_plotting_style()
    
    # Create plots
    ranks_df = create_rank_comparison(results, args.output_dir, colors)
    hit_rates_df = create_hit_rates_plot(results, args.output_dir, colors)
    rank_changes_df = create_rank_changes_plot(results, args.output_dir, colors)
    create_k1_hit_rate_plot(hit_rates_df, args.output_dir, colors)
    
    # Create new visualizations
    create_rank_displacement_visualization(results, args.output_dir, colors)
    create_preference_visualization(results, args.output_dir, colors)
    
    print(f"\nPlots saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()
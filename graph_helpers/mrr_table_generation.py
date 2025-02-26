import json
import argparse
import pandas as pd
from pathlib import Path
import os

def read_json_results(file_path):
    """Read experiment results from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def create_mrr_tables(json_files, output_dir, output_format='latex'):
    """
    Create MRR comparison tables from experiment result files.
    
    Args:
        json_files (list): List of JSON file paths
        output_dir (str): Directory to save the output tables
        output_format (str): Format of the output tables ('latex', 'markdown', or 'csv')
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect data from all files
    all_data = []
    
    for file_path in json_files:
        data = read_json_results(file_path)
        
        # Determine if this is reranking or pseudocode experiment
        experiment_type = 'unknown'
        sample_entry = data[0] if isinstance(data, list) else data
        
        if 'reranked_mrr' in sample_entry:
            experiment_type = 'reranking'
            mrr_keys = ('baseline_mrr', 'reranked_mrr')
        elif 'pseudo_mrr' in sample_entry:
            experiment_type = 'pseudocode'
            mrr_keys = ('baseline_mrr', 'pseudo_mrr')
        else:
            print(f"Warning: Could not determine experiment type for {file_path}")
            continue
        
        # Process data based on experiment type
        if isinstance(data, list):
            # Extract relevant data for each entry
            for entry in data:
                if all(k in entry for k in mrr_keys):
                    result = {
                        'llm_model': entry.get('llm_model', 'Unknown'),
                        'embedding_model': entry.get('embedding_model', 'Unknown'),
                        'normalization_type': entry.get('normalization_type', 'Unknown'),
                        'baseline_mrr': entry.get(mrr_keys[0], 0),
                        'enhanced_mrr': entry.get(mrr_keys[1], 0),
                        'experiment_type': experiment_type
                    }
                    
                    # Add experiment-specific fields
                    if experiment_type == 'reranking':
                        result['rerank_k'] = entry.get('rerank_k', 0)
                        result['alpha'] = entry.get('alpha', 0)
                    elif experiment_type == 'pseudocode':
                        result['is_pseudo'] = entry.get('is_pseudo', False)
                    
                    all_data.append(result)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(all_data)
    
    # Create tables grouped by different criteria
    if not df.empty:
        # Group by LLM model and normalization type
        grouped_tables = {}
        
        # For reranking experiments
        if 'reranking' in df['experiment_type'].values:
            rerank_df = df[df['experiment_type'] == 'reranking']
            
            # Group by llm_model and embedding_model
            for (llm, embedding), group in rerank_df.groupby(['llm_model', 'embedding_model']):
                key = f"reranking_{llm.split('/')[-1]}_{embedding.split('/')[-1]}"
                grouped_tables[key] = group.pivot_table(
                    index='normalization_type',
                    columns='rerank_k',
                    values=['baseline_mrr', 'enhanced_mrr'],
                    aggfunc='mean'
                )
        
        # For pseudocode experiments
        if 'pseudocode' in df['experiment_type'].values:
            pseudo_df = df[df['experiment_type'] == 'pseudocode']
            
            # Group by llm_model and embedding_model
            for (llm, embedding), group in pseudo_df.groupby(['llm_model', 'embedding_model']):
                key = f"pseudocode_{llm.split('/')[-1]}_{embedding.split('/')[-1]}"
                grouped_tables[key] = group.pivot_table(
                    index='normalization_type',
                    columns='is_pseudo',
                    values=['baseline_mrr', 'enhanced_mrr'],
                    aggfunc='mean'
                )
        
        # Generate tables in the requested format
        for name, table_df in grouped_tables.items():
            output_file = output_dir / f"{name}_mrr_table"
            
            if output_format == 'latex':
                with open(f"{output_file}.tex", 'w') as f:
                    latex_table = generate_latex_table(table_df, name)
                    f.write(latex_table)
                
            elif output_format == 'markdown':
                with open(f"{output_file}.md", 'w') as f:
                    markdown_table = table_df.to_markdown()
                    f.write(f"# MRR Table for {name}\n\n{markdown_table}\n")
                
            elif output_format == 'csv':
                table_df.to_csv(f"{output_file}.csv")
            
            print(f"Created {output_format} table for {name}")
    else:
        print("No valid data found to create tables")

def generate_latex_table(df, table_name):
    """Generate a LaTeX table from a DataFrame."""
    # Get column and index information
    cols = df.columns.tolist()
    idx = df.index.tolist()
    
    # Start building LaTeX table
    latex = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\begin{tabular}{l" + "r" * len(cols) + "}",
        "\\toprule"
    ]
    
    # Add header row
    header = ["Normalization"]
    for col in cols:
        metric, config = col
        if metric == 'baseline_mrr':
            header.append(f"Baseline")
        else:
            if isinstance(config, bool):  # pseudocode
                header.append("Pseudocode" if config else "Direct")
            else:  # reranking
                header.append(f"Reranked (k={config})")
    
    latex.append(" & ".join(header) + " \\\\")
    latex.append("\\midrule")
    
    # Add data rows
    for norm_type in idx:
        row = [norm_type]
        for col in cols:
            val = df.loc[norm_type, col]
            row.append(f"{val:.3f}")
        latex.append(" & ".join(row) + " \\\\")
    
    # Close the table
    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{MRR comparison for {table_name}}}",
        f"\\label{{tab:mrr_{table_name.lower().replace('-', '_')}}}",
        "\\end{table}"
    ])
    
    return "\n".join(latex)

if __name__ == "__main__":
    
    input_files = [
        './results/humaneval_metrics/humaneval_reranker_metrics_20250225_130238_results.json'
        # './results/mbpp_metrics/mbpp_reranker_metrics_20250225_134929_results.json'
    ]
    
    create_mrr_tables(input_files, './', 'latex')

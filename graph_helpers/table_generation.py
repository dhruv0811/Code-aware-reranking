import json

def create_latex_tables(json_file, output_file):
    # Read JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Filter for rerank_k = 25
    try:
        filtered_data = [entry for entry in data if entry['rerank_k'] == 25]
    except Exception as e:
        print("WARNING: COULD NOT FILTER DATA!")
        filtered_data = data
    
    # Group data by model combinations
    model_groups = {}
    for entry in filtered_data:
        if entry['llm_model'] == 'meta-llama/Llama-3.1-8B-Instruct':
            model_key = (entry['llm_model'], entry['embedding_model'])
            if model_key not in model_groups:
                model_groups[model_key] = []
            model_groups[model_key].append(entry)
    
    # Define custom ordering for normalization types
    order = {'none': 0, 'docstring': 1, 'functions': 2, 'variables': 3, 'both': 4}
    name_mapping = {
        'none': 'None',
        'docstring': 'Docstring',
        'functions': 'Functions',
        'variables': 'Variables',
        'both': 'Both'
    }
    
    # Start the LaTeX document
    latex_tables = [
        "% Required packages: booktabs, xcolor, colortbl",
        "% Define colors for different levels of change",
        "\\definecolor{lightgreen}{RGB}{220,240,220}",
        "\\definecolor{darkgreen}{RGB}{150,200,150}",
        "\\definecolor{lightred}{RGB}{240,220,220}",
        "\\definecolor{darkred}{RGB}{200,150,150}",
        ""
    ]
    
    # Create a table for each model combination
    for (llm_model, embedding_model), group_data in model_groups.items():
        # Sort the data by normalization type
        group_data.sort(key=lambda x: order[x['normalization_type']])
        
        recall_ks = [1, 5, 10, 25]
        
        # Build each table
        table = [
            "\\begin{table}[h]",
            "\\centering",
            "\\begin{tabular}{l|" + "c" * (len(recall_ks) - 1) + "|c}",
            "\\toprule",
            "\\textbf{Normalization} & " + " & ".join([f"\\textbf{{Recall@{k}}}" for k in recall_ks]) + " \\\\",
            "\\midrule"
        ]
        
        # Add data rows
        for entry in group_data:
            norm_type = name_mapping[entry['normalization_type']]
            scores = []
            for k in recall_ks:
                baseline = entry[f'baseline_recall@{k}']
                reranked = entry[f'reranked_recall@{k}']
                # reranked = entry[f'pseudo_recall@{k}']
                
                # Handle Recall@25 differently
                if k == 25:
                    score_str = f"{baseline*100:.1f}/-"
                else:
                    score_str = f"{baseline*100:.1f}/{reranked*100:.1f}"
                
                # Calculate difference in percentage points
                diff = (reranked - baseline) * 100
                
                # Add cell color based on difference (skip for Recall@50)
                if k != 25:  # Only color non-Recall@50 cells
                    if diff > 10:
                        score_str = f"\\cellcolor{{darkgreen}}{score_str}"
                    elif diff > 0:
                        score_str = f"\\cellcolor{{lightgreen}}{score_str}"
                    elif diff < -10:
                        score_str = f"\\cellcolor{{darkred}}{score_str}"
                    elif diff < 0:
                        score_str = f"\\cellcolor{{lightred}}{score_str}"
                
                scores.append(score_str)
            
            row = f"{norm_type} & " + " & ".join(scores) + " \\\\"
            table.append(row)
        
        # Close the table
        table.extend([
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{Baseline/Pseudocode Recall Scores (\\%) using {llm_model} and {embedding_model}. " +
            "Dark green: $>$10 points improvement, Light green: 0-10 points improvement, " +
            "Light red: 0-10 points decrease, Dark red: $>$10 points decrease. " +
            "For Recall@25, only baseline scores are shown.}",
            "\\label{tab:pseudo-scores-" + llm_model.split('/')[-1].lower().replace('-', '_') + "}",
            "\\end{table}",
            ""
        ])
        
        latex_tables.extend(table)
    
    print(latex_tables)
    
    # Save all tables to file
    with open(output_file, 'w') as f:
        f.write("\n".join(latex_tables))
    
    return "\n".join(latex_tables)

# Example usage
if __name__ == "__main__":
    # latex_output = create_latex_tables("all_results_fixed_corpus/experiment_redo3_20241207_234604_results.json", 
    #                                    "tables/FIXED_all_model_results.tex")
    # print("Tables have been saved to tables/all_model_results.tex")

    # latex_output = create_latex_tables("pseduo_merged_results.json", 
    #                                    "tables/pseudo_all_model_results.tex")
    
    latex_output = create_latex_tables("./results/fixed_corpus_mbpp_reranker/experiment_redo3_20250212_130621_results.json", 
                                       "mbpp_reranker_results.tex")
    
    # print("Tables have been saved to tables/all_model_results.tex")

# Example usage
# if __name__ == "__main__":
#     latex_output = create_latex_table("humaneval_rerank_alpha_0.7_base/experiment_20241206_221605_results.json",
#                                       "tables/humaneval_rerank_alpha_0.7_base.tex")
#     latex_output = create_latex_table("humaneval_rerank_alpha_0.7_large/experiment_20241206_184828_results.json",
#                                       "tables/humaneval_rerank_alpha_0.7_large.tex")
    
#     latex_output = create_latex_table("mixtral_rerank_0.7_base/experiment_20241206_231822_results.json",
#                                       "tables/mixtral_rerank_alpha_0.7_base.tex")
#     latex_output = create_latex_table("mixtral_rerank_0.7_large/experiment_20241206_215734_results.json",
#                                       "tables/mixtral_rerank_alpha_0.7_large.tex")

#     print(latex_output)
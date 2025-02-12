import json
from collections import defaultdict

def merge_results(json_data):
    """
    Merge JSON entries with the same model names and normalization types,
    using baseline results when is_pseudo=False and reranked results when is_pseudo=True.
    
    Args:
        json_data (list): List of dictionaries containing the results
        
    Returns:
        list: List of merged dictionaries
    """
    # Create a dictionary to group entries by model and normalization type
    grouped_data = defaultdict(list)
    for entry in json_data:
        key = (entry['llm_model'], entry['embedding_model'], entry['normalization_type'])
        grouped_data[key].append(entry)
    
    merged_results = []
    
    # Process each group
    for key, entries in grouped_data.items():
        if len(entries) < 2:
            continue
            
        # Find the baseline and pseudo entries
        baseline_entry = None
        pseudo_entry = None
        for entry in entries:
            if entry['is_pseudo']:
                pseudo_entry = entry
            else:
                baseline_entry = entry
                
        if baseline_entry and pseudo_entry:
            # Create merged entry
            merged_entry = {
                'llm_model': key[0],
                'embedding_model': key[1],
                'normalization_type': key[2]
            }
            
            # Add baseline results
            for metric in [k for k in baseline_entry.keys() if k.startswith('baseline_recall@')]:
                merged_entry[metric] = baseline_entry[metric]
                
            # Add reranked results (pseudo results)
            for metric in [k for k in pseudo_entry.keys() if k.startswith('pseudo_recall@')]:
                merged_entry[metric] = pseudo_entry[metric]
                
            merged_results.append(merged_entry)
    
    return merged_results

# Example usage:
if __name__ == "__main__":
    # Read JSON file
    with open('./results/mbpp_pseudocode/experiment_redo3_20250212_130553_results.json', 'r') as f:
        data = json.load(f)
    
    # Merge results
    merged_data = merge_results(data)
    
    # Write merged results to new file
    with open('./results/mbpp_pseudocode/pseduo_merged_results.json', 'w') as f:
        json.dump(merged_data, f, indent=2)
        
    print(f"Successfully merged {len(merged_data)} entries")
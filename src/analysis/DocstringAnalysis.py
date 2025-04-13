import os
import re
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any
import re

class DocstringBiasAnalyzer:
    """
    Analyzer for docstring bias experiment results.
    Provides detailed analysis and visualization of experiment outcomes.
    Uses k+1 rank for gold documents not found in the results.
    """
    
    def __init__(self, results_dir: str = "/home/gganeshl/Code-aware-reranking/results/docstring_bias"):
        """Initialize the analyzer with the directory containing experiment results."""
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            raise ValueError(f"Results directory does not exist: {results_dir}")
        
        self.raw_results = None
        self.norm_results = None
        self.combined_df = None
    
    def load_experiment_results(self):
        """Load the most recent raw and normalized experiment results."""
        # Find the most recent experiment files
        raw_files = list(self.results_dir.glob("docstring_bias_raw_*.json"))
        norm_files = list(self.results_dir.glob("docstring_bias_normalized_*.json"))
        
        if not raw_files or not norm_files:
            raise ValueError("Could not find experiment result files")
        
        # Sort by modification time (most recent first)
        raw_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        norm_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Load the most recent results
        with open(raw_files[0], 'r') as f:
            self.raw_results = json.load(f)
            print(f"Loaded raw results: {raw_files[0].name}")
        
        with open(norm_files[0], 'r') as f:
            self.norm_results = json.load(f)
            print(f"Loaded normalized results: {norm_files[0].name}")
        
        return self.raw_results, self.norm_results
    
    def create_combined_dataframe(self):
        """Create a combined DataFrame from all experiment results for easier analysis."""
        if not self.raw_results or not self.norm_results:
            self.load_experiment_results()
        
        rows = []
        
        # Process raw results
        for model_result in self.raw_results["model_results"]:
            base_row = {
                "embedding_model": model_result["embedding_model"],
                "normalization": "none",
                "avg_rank_with_docstring": model_result["avg_with_docstring_rank"],
                "avg_rank_without_docstring": model_result["avg_without_docstring_rank"],
                "median_rank_with_docstring": model_result["median_with_docstring_rank"],
                "median_rank_without_docstring": model_result["median_without_docstring_rank"],
                "rank_improvement": model_result["rank_improvement"]
            }
            
            # Add hit rates
            for k, rate in model_result["with_docstring_hit_rates"].items():
                base_row[f"R@{k}_with_docstring"] = rate
            
            for k, rate in model_result["without_docstring_hit_rates"].items():
                base_row[f"R@{k}_without_docstring"] = rate
                base_row[f"R@{k}_difference"] = model_result["with_docstring_hit_rates"][k] - rate
            
            rows.append(base_row)
        
        # Process normalized results
        for model_result in self.norm_results["model_results"]:
            base_row = {
                "embedding_model": model_result["embedding_model"],
                "normalization": "docstring",
                "avg_rank_with_docstring": model_result["avg_with_docstring_rank"],
                "avg_rank_without_docstring": model_result["avg_without_docstring_rank"],
                "median_rank_with_docstring": model_result["median_with_docstring_rank"],
                "median_rank_without_docstring": model_result["median_without_docstring_rank"],
                "rank_improvement": model_result["rank_improvement"]
            }
            
            # Add hit rates
            for k, rate in model_result["with_docstring_hit_rates"].items():
                base_row[f"R@{k}_with_docstring"] = rate
            
            for k, rate in model_result["without_docstring_hit_rates"].items():
                base_row[f"R@{k}_without_docstring"] = rate
                base_row[f"R@{k}_difference"] = model_result["with_docstring_hit_rates"][k] - rate
            
            rows.append(base_row)
        
        self.combined_df = pd.DataFrame(rows)
        return self.combined_df
    
    def analyze_results(self):
        """Perform comprehensive analysis of experiment results."""
        if self.combined_df is None:
            self.create_combined_dataframe()
        
        print("\n=== DOCSTRING BIAS ANALYSIS ===\n")
        
        # 1. Overall average rank comparison
        print("Average Rank Comparison (lower is better):")
        avg_ranks = self.combined_df.groupby('normalization')[
            ['avg_rank_with_docstring', 'avg_rank_without_docstring', 'rank_improvement']
        ].mean()
        print(avg_ranks)
        
        # 2. Per-model analysis
        print("\nPer-Model Analysis:")
        for model in self.combined_df['embedding_model'].unique():
            model_data = self.combined_df[self.combined_df['embedding_model'] == model]
            print(f"\n{model}:")
            for norm_type in ['none', 'docstring']:
                norm_data = model_data[model_data['normalization'] == norm_type]
                if not norm_data.empty:
                    print(f"  {norm_type.capitalize()} normalization:")
                    print(f"    With docstring avg rank: {norm_data['avg_rank_with_docstring'].values[0]:.2f}")
                    print(f"    Without docstring avg rank: {norm_data['avg_rank_without_docstring'].values[0]:.2f}")
                    print(f"    Rank improvement: {norm_data['rank_improvement'].values[0]:.2f}")
        
        # 3. Hit rate analysis at different k values
        print("\nHit Rate Analysis (R@k):")
        k_values = [1, 5, 10, 25, 50, 100]
        
        for k in k_values:
            if f"R@{k}_with_docstring" in self.combined_df.columns:
                print(f"\nR@{k} Analysis:")
                r_at_k = self.combined_df.groupby('normalization')[
                    [f"R@{k}_with_docstring", f"R@{k}_without_docstring", f"R@{k}_difference"]
                ].mean()
                print(r_at_k)
        
        # 4. Normalization effect analysis
        print("\nNormalization Effect Analysis:")
        
        # Prepare the data
        norm_effect = []
        for model in self.combined_df['embedding_model'].unique():
            model_data = self.combined_df[self.combined_df['embedding_model'] == model]
            
            # Get raw and normalized data for this model
            raw_data = model_data[model_data['normalization'] == 'none']
            norm_data = model_data[model_data['normalization'] == 'docstring']
            
            if not raw_data.empty and not norm_data.empty:
                # Calculate the effect of normalization
                with_doc_effect = raw_data['avg_rank_with_docstring'].values[0] - norm_data['avg_rank_with_docstring'].values[0]
                without_doc_effect = raw_data['avg_rank_without_docstring'].values[0] - norm_data['avg_rank_without_docstring'].values[0]
                
                norm_effect.append({
                    'embedding_model': model,
                    'with_docstring_effect': with_doc_effect,
                    'without_docstring_effect': without_doc_effect,
                    'difference': with_doc_effect - without_doc_effect
                })
        
        norm_effect_df = pd.DataFrame(norm_effect)
        print(norm_effect_df)
        
        return {
            'overall_avg_ranks': avg_ranks,
            'normalization_effect': norm_effect_df
        }
    
    def plot_results(self, output_dir: str = None):
        """Generate visualizations of experiment results."""
        if self.combined_df is None:
            self.create_combined_dataframe()
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = self.results_dir / "plots"
            output_path.mkdir(exist_ok=True)
        
        # Set plotting style
        sns.set(style="whitegrid")
        plt.rcParams.update({'font.size': 12})
        
        # 1. Average Rank Comparison
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        plot_data = []
        for _, row in self.combined_df.iterrows():
            plot_data.append({
                'model': row['embedding_model'].split('/')[-1],
                'normalization': row['normalization'],
                'rank_type': 'With Docstring',
                'avg_rank': row['avg_rank_with_docstring']
            })
            plot_data.append({
                'model': row['embedding_model'].split('/')[-1],
                'normalization': row['normalization'],
                'rank_type': 'Without Docstring',
                'avg_rank': row['avg_rank_without_docstring']
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create the grouped bar chart
        ax = sns.barplot(
            data=plot_df, 
            x='model', 
            y='avg_rank', 
            hue='rank_type',
            palette='Set2',
            errorbar=None
        )
        
        # Customize the plot
        plt.title('Average Rank Comparison: With vs. Without Docstrings', fontsize=14)
        plt.xlabel('Embedding Model', fontsize=12)
        plt.ylabel('Average Rank (lower is better)', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Code Type')
        
        # Add facet for normalization
        facet = sns.FacetGrid(plot_df, col='normalization', height=6, aspect=1.2)
        facet.map_dataframe(sns.barplot, x='model', y='avg_rank', hue='rank_type', palette='Set2', errorbar=None)
        facet.add_legend()
        facet.set_axis_labels('Embedding Model', 'Average Rank (lower is better)')
        facet.set_titles(col_template='{col_name} normalization')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_path / 'avg_rank_comparison.png', dpi=300, bbox_inches='tight')
        
        # 2. Rank Improvement Analysis
        plt.figure(figsize=(10, 6))
        
        # Filter and prepare data
        improvement_data = self.combined_df[['embedding_model', 'normalization', 'rank_improvement']].copy()
        improvement_data['model'] = improvement_data['embedding_model'].apply(lambda x: x.split('/')[-1])
        
        # Create the bar chart
        ax = sns.barplot(
            data=improvement_data, 
            x='model', 
            y='rank_improvement', 
            hue='normalization',
            palette='Set1',
            errorbar=None
        )
        
        # Customize the plot
        plt.title('Rank Improvement from Using Docstrings', fontsize=14)
        plt.xlabel('Embedding Model', fontsize=12)
        plt.ylabel('Average Rank Improvement', fontsize=12)
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        plt.legend(title='Normalization')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_path / 'rank_improvement.png', dpi=300, bbox_inches='tight')
        
        # 3. Hit Rate Comparison (R@k)
        k_values = [1, 5, 10, 25, 50, 100]
        
        plt.figure(figsize=(12, 8))
        
        # Prepare data for plotting
        hit_rate_data = []
        for _, row in self.combined_df.iterrows():
            model_name = row['embedding_model'].split('/')[-1]
            for k in k_values:
                if f"R@{k}_with_docstring" in self.combined_df.columns:
                    hit_rate_data.append({
                        'model': model_name,
                        'normalization': row['normalization'],
                        'k_value': k,
                        'document_type': 'With Docstring',
                        'hit_rate': row[f"R@{k}_with_docstring"]
                    })
                    hit_rate_data.append({
                        'model': model_name,
                        'normalization': row['normalization'],
                        'k_value': k,
                        'document_type': 'Without Docstring',
                        'hit_rate': row[f"R@{k}_without_docstring"]
                    })
        
        hit_rate_df = pd.DataFrame(hit_rate_data)
        
        # Create separate plots for each k value
        for k in k_values:
            plt.figure(figsize=(12, 6))
            k_data = hit_rate_df[hit_rate_df['k_value'] == k]
            
            if k_data.empty:
                continue
                
            # Create the grouped bar chart
            ax = sns.barplot(
                data=k_data,
                x='model',
                y='hit_rate',
                hue='document_type',
                palette='Set2',
                errorbar=None
            )
            
            # Customize the plot
            plt.title(f'Hit Rate Comparison at R@{k}', fontsize=14)
            plt.xlabel('Embedding Model', fontsize=12)
            plt.ylabel(f'Hit Rate (R@{k})', fontsize=12)
            plt.xticks(rotation=45)
            plt.legend(title='Code Type')
            
            # Create facet for normalization
            facet = sns.FacetGrid(k_data, col='normalization', height=6, aspect=1.2)
            facet.map_dataframe(sns.barplot, x='model', y='hit_rate', hue='document_type', palette='Set2', errorbar=None)
            facet.add_legend()
            facet.set_axis_labels('Embedding Model', f'Hit Rate (R@{k})')
            facet.set_titles(col_template='{col_name} normalization')
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(output_path / f'hit_rate_R{k}.png', dpi=300, bbox_inches='tight')
        
        # 4. Combined hit rate plot
        plt.figure(figsize=(14, 8))
        
        # Create a line plot showing hit rate trends across k values
        sns.lineplot(
            data=hit_rate_df,
            x='k_value',
            y='hit_rate',
            hue='document_type',
            style='normalization',
            markers=True,
            dashes=True
        )
        
        plt.title('Hit Rate Performance Across Different k Values', fontsize=14)
        plt.xlabel('k Value', fontsize=12)
        plt.ylabel('Hit Rate', fontsize=12)
        plt.legend(title='Configuration')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_path / 'hit_rate_trends.png', dpi=300, bbox_inches='tight')
        
        # 5. Normalization Effect Analysis
        plt.figure(figsize=(10, 6))
        
        # Prepare the data for normalization effect
        norm_effect = []
        for model in self.combined_df['embedding_model'].unique():
            model_data = self.combined_df[self.combined_df['embedding_model'] == model]
            
            # Get raw and normalized data for this model
            raw_data = model_data[model_data['normalization'] == 'none']
            norm_data = model_data[model_data['normalization'] == 'docstring']
            
            if not raw_data.empty and not norm_data.empty:
                # Calculate the effect of normalization
                with_doc_effect = raw_data['avg_rank_with_docstring'].values[0] - norm_data['avg_rank_with_docstring'].values[0]
                without_doc_effect = raw_data['avg_rank_without_docstring'].values[0] - norm_data['avg_rank_without_docstring'].values[0]
                
                norm_effect.append({
                    'model': model.split('/')[-1],
                    'with_docstring_effect': with_doc_effect,
                    'without_docstring_effect': without_doc_effect
                })
        
        norm_effect_df = pd.DataFrame(norm_effect)
        
        # Reshape for grouped bar chart
        plot_data = []
        for _, row in norm_effect_df.iterrows():
            plot_data.append({
                'model': row['model'],
                'effect_type': 'With Docstring',
                'rank_improvement': row['with_docstring_effect']
            })
            plot_data.append({
                'model': row['model'],
                'effect_type': 'Without Docstring',
                'rank_improvement': row['without_docstring_effect']
            })
        
        effect_plot_df = pd.DataFrame(plot_data)
        
        # Create the grouped bar chart
        ax = sns.barplot(
            data=effect_plot_df,
            x='model',
            y='rank_improvement',
            hue='effect_type',
            palette='Set1',
            errorbar=None
        )
        
        # Customize the plot
        plt.title('Effect of Normalization on Document Ranking', fontsize=14)
        plt.xlabel('Embedding Model', fontsize=12)
        plt.ylabel('Rank Improvement (positive is better)', fontsize=12)
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        plt.legend(title='Document Type')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_path / 'normalization_effect.png', dpi=300, bbox_inches='tight')
        
        print(f"All plots saved to {output_path}")
        
        return {
            'output_path': output_path,
            'plots': [
                'avg_rank_comparison.png',
                'rank_improvement.png',
                'hit_rate_trends.png',
                'normalization_effect.png'
            ]
        }
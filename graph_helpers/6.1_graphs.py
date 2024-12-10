import matplotlib.pyplot as plt
import numpy as np

# Data preparation
k_values = [1, 5, 10, 25]
models = {
    'Mixtral-8x7B': {
        'None': {'baseline': [100.0, 100.0, 100.0, 100.0], 'reranked': [99.4, 100.0, 100.0, None]},
        'Docstring': {'baseline': [98.2, 100.0, 100.0, 100.0], 'reranked': [98.8, 100.0, 100.0, None]},
        'Functions': {'baseline': [36.0, 60.4, 73.8, 82.3], 'reranked': [50.6, 75.6, 80.5, None]},
        'Variables': {'baseline': [75.0, 90.2, 95.1, 97.0], 'reranked': [86.6, 96.3, 97.0, None]},
        'Both': {'baseline': [4.9, 11.0, 12.2, 21.3], 'reranked': [11.0, 20.1, 21.3, None]}
    },
    'Llama-3.1-8B': {
        'None': {'baseline': [100.0, 100.0, 100.0, 100.0], 'reranked': [99.4, 100.0, 100.0, None]},
        'Docstring': {'baseline': [98.2, 100.0, 100.0, 100.0], 'reranked': [97.6, 99.4, 100.0, None]},
        'Functions': {'baseline': [36.0, 60.4, 73.8, 82.3], 'reranked': [53.7, 78.7, 81.1, None]},
        'Variables': {'baseline': [75.0, 90.2, 95.1, 97.0], 'reranked': [84.1, 95.1, 97.0, None]},
        'Both': {'baseline': [4.9, 11.0, 12.2, 21.3], 'reranked': [11.0, 20.7, 21.3, None]}
    },
    'Llama-3.1-70B': {
        'None': {'baseline': [100.0, 100.0, 100.0, 100.0], 'reranked': [100.0, 100.0, 100.0, None]},
        'Docstring': {'baseline': [98.2, 100.0, 100.0, 100.0], 'reranked': [97.6, 100.0, 100.0, None]},
        'Functions': {'baseline': [36.0, 60.4, 73.8, 82.3], 'reranked': [56.1, 79.9, 81.1, None]},
        'Variables': {'baseline': [75.0, 90.2, 95.1, 97.0], 'reranked': [80.5, 96.3, 97.0, None]},
        'Both': {'baseline': [4.9, 11.0, 12.2, 21.3], 'reranked': [12.8, 20.1, 21.3, None]}
    }
}

# Define y-axis ranges for each normalization type
y_ranges = {
    'Docstring': (95, 101),  # Docstring values are between 97-100
    'Functions': (45, 100),   # Functions values are between 36-82
    'Variables': (45, 100),  # Variables values are between 75-97
    'Both': (5, 25)         # Both values are between 4-21
}

# Create the plot
plt.style.use('seaborn')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
axes = {'Functions': ax1, 'Variables': ax2}
colors = {'Mixtral-8x7B': 'blue', 'Llama-3.1-8B': 'red', 'Llama-3.1-70B': 'green'}
k_plot = [1, 5, 10]

for norm_type, ax in axes.items():
    # Plot baseline Recall@25 as horizontal line
    baseline_25 = models['Mixtral-8x7B'][norm_type]['baseline'][3]
    ax.axhline(y=baseline_25, color='gray', linestyle='--', label='Baseline Recall@25')
    
    for model_name, color in colors.items():
        reranked = [models[model_name][norm_type]['reranked'][i] for i in range(3)]
        ax.plot(k_plot, reranked, color=color, linestyle='--', marker='s', label=f'{model_name} (Reranked)')
    
    ax.set_xlabel('k')
    ax.set_ylabel('Recall@k (%)')
    ax.set_title(f'Corpus Normalization: {norm_type}')
    ax.grid(True)
    ax.set_xticks(k_plot)
    ax.set_ylim(y_ranges[norm_type])

# Add legend to the bottom right of the second plot
handles, labels = ax1.get_legend_handles_labels()
legend = ax2.legend(handles, labels, loc='lower right', 
                   frameon=True, 
                   facecolor='white', 
                   edgecolor='black',
                   framealpha=0.8)

plt.tight_layout()
plt.suptitle('Recall Performance Comparison Across Models and Normalization Types', y=1.05, fontsize=14)

output_path = "6.1_plots.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
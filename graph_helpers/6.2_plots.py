import matplotlib.pyplot as plt
import numpy as np

# Define k values
k_values = [1, 5, 10]

# Data for Functions normalization
llama_70b_base_func = [
    (49.4 - 31.1) / 76.2 * 100,  # @1
    (72.6 - 55.5) / 76.2 * 100,  # @5
    (75.6 - 62.2) / 76.2 * 100,  # @10
]

llama_70b_large_func = [
    (56.1 - 36.0) / 82.3 * 100,  # @1
    (79.9 - 60.4) / 82.3 * 100,  # @5
    (81.1 - 73.8) / 82.3 * 100,  # @10
]

llama_8b_base_func = [
    (47.6 - 31.1) / 76.2 * 100,  # @1
    (71.3 - 55.5) / 76.2 * 100,  # @5
    (75.6 - 62.2) / 76.2 * 100,  # @10
]

llama_8b_large_func = [
    (50.6 - 36.0) / 82.3 * 100,  # @1
    (75.6 - 60.4) / 82.3 * 100,  # @5
    (80.5 - 73.8) / 82.3 * 100,  # @10
]

mixtral_base_func = [
    (44.5 - 31.1) / 76.2 * 100,  # @1
    (69.5 - 55.5) / 76.2 * 100,  # @5
    (74.4 - 62.2) / 76.2 * 100,  # @10
]

mixtral_large_func = [
    (53.7 - 36.0) / 82.3 * 100,  # @1
    (78.7 - 60.4) / 82.3 * 100,  # @5
    (81.1 - 73.8) / 82.3 * 100,  # @10
]

# Data for Variables normalization
llama_70b_base_var = [
    (89.0 - 82.3) / 98.8 * 100,  # @1
    (97.6 - 95.7) / 98.8 * 100,  # @5
    (98.8 - 97.6) / 98.8 * 100,  # @10
]

llama_70b_large_var = [
    (80.5 - 75.0) / 97.0 * 100,  # @1
    (96.3 - 90.2) / 97.0 * 100,  # @5
    (97.0 - 95.1) / 97.0 * 100,  # @10
]

llama_8b_base_var = [
    (83.5 - 82.3) / 98.8 * 100,  # @1
    (97.6 - 95.7) / 98.8 * 100,  # @5
    (98.2 - 97.6) / 98.8 * 100,  # @10
]

llama_8b_large_var = [
    (86.6 - 75.0) / 97.0 * 100,  # @1
    (96.3 - 90.2) / 97.0 * 100,  # @5
    (97.0 - 95.1) / 97.0 * 100,  # @10
]

mixtral_base_var = [
    (89.6 - 82.3) / 98.8 * 100,  # @1
    (98.2 - 95.7) / 98.8 * 100,  # @5
    (98.8 - 97.6) / 98.8 * 100,  # @10
]

mixtral_large_var = [
    (84.1 - 75.0) / 97.0 * 100,  # @1
    (95.1 - 90.2) / 97.0 * 100,  # @5
    (97.0 - 95.1) / 97.0 * 100,  # @10
]

# Create the plot
plt.style.use('seaborn')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Define colors and line styles
base_color = '#2E86C1'    # Blue for base models
large_color = '#E74C3C'   # Red for large models
line_styles = ['-', '--', '-.']  # Different line styles for different model families

# Functions plot
ax1.plot(k_values, llama_70b_base_func, line_styles[0], color=base_color, label=' GIST-Base, Llama-70B', marker='o')
ax1.plot(k_values, llama_8b_base_func, line_styles[1], color=base_color, label='GIST-Base, Llama-8B', marker='s')
ax1.plot(k_values, mixtral_base_func, line_styles[2], color=base_color, label='GIST-Base, Mixtral8x7B', marker='^')

ax1.plot(k_values, llama_70b_large_func, line_styles[0], color=large_color, label='GIST-Large, Llama-70B', marker='o')
ax1.plot(k_values, llama_8b_large_func, line_styles[1], color=large_color, label='GIST-Large, Llama-8B', marker='s')
ax1.plot(k_values, mixtral_large_func, line_styles[2], color=large_color, label='GIST-Large, Mixtral8x7B', marker='^')

# Variables plot
ax2.plot(k_values, llama_70b_base_var, line_styles[0], color=base_color, label='GIST-Base, Llama-70B', marker='o')
ax2.plot(k_values, llama_8b_base_var, line_styles[1], color=base_color, label='GIST-Base, Llama-8B', marker='s')
ax2.plot(k_values, mixtral_base_var, line_styles[2], color=base_color, label='GIST-Base, Mixtral8x7B', marker='^')

ax2.plot(k_values, llama_70b_large_var, line_styles[0], color=large_color, label='GIST-Large, Llama-70B', marker='o')
ax2.plot(k_values, llama_8b_large_var, line_styles[1], color=large_color, label='GIST-Large, Llama-8B', marker='s')
ax2.plot(k_values, mixtral_large_var, line_styles[2], color=large_color, label='GIST-Large, Mixtral8x7B', marker='^')

# Customize the plots
for ax, title in zip([ax1, ax2], ['Functions Normalization', 'Variables Normalization']):
    ax.set_xlabel('Recall@k')
    ax.set_ylabel('Relative Improvement (%)')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(k_values)

ax2.legend(loc='upper right', 
                   frameon=True, 
                   facecolor='white', 
                   edgecolor='black',
                   framealpha=0.8)

plt.tight_layout()
plt.suptitle('Relative Improvement in Recall Across Embedding Model Sizes', y=1.05)

output_path = "6.2_plots.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# Show the plot
# plt.show()
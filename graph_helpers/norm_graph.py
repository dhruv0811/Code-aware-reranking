import matplotlib.pyplot as plt

# Baseline recall scores from the table
normalization_methods = ["None", "Docstring", "Variables", "Functions", "Both"]
recall_at_1 = [100.0, 98.2, 75.0, 36.0, 4.9]
recall_at_5 = [100.0, 100.0, 90.2, 60.4, 11.0]
recall_at_10 = [100.0, 100.0, 95.1, 73.8, 12.2]
recall_at_25 = [100.0, 100.0, 97.0, 82.3, 21.3]

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(normalization_methods, recall_at_1, marker='o', label="Recall@1")
plt.plot(normalization_methods, recall_at_5, marker='o', label="Recall@5")
plt.plot(normalization_methods, recall_at_10, marker='o', label="Recall@10")
plt.plot(normalization_methods, recall_at_25, marker='o', label="Recall@25")

# Adding labels, legend, and title
plt.xlabel("Normalization Methods")
plt.ylabel("Baseline Recall Scores (%)")
plt.title("Baseline Recall Scores with Normalization")
plt.legend()
plt.grid(visible=True, linestyle="--", alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot as a PNG file
output_path = "baseline_recall_scores.png"
plt.savefig(output_path, dpi=300)

# Show the plot
plt.show()

print(f"Plot saved as '{output_path}'.")

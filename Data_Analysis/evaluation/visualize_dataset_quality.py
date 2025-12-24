# visualize_data_quality.py

"""
Visualize the data quality comparison result.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load stats
df = pd.read_csv("data_quality_comparison_3stage.csv")

stages = df["stage"]

# Plot 1: Number of papers
plt.figure()
plt.bar(stages, df["num_papers"])
plt.title("Number of Papers Across Processing Stages")
plt.ylabel("Count")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("num_papers_comparison.png")
plt.close()

# Plot 2: Schema completeness
plt.figure()
plt.bar(stages, df["schema_completeness_%"])
plt.title("Schema Completeness (%)")
plt.ylabel("Percentage")
plt.ylim(0, 100)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("schema_completeness.png")
plt.close()

# Plot 3: Abstract coverage
plt.figure()
plt.bar(stages, df["has_abstract_%"])
plt.title("Abstract Coverage (%)")
plt.ylabel("Percentage")
plt.ylim(0, 100)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("abstract_coverage.png")
plt.close()

# Plot 4: Quality score (if exists)
if "avg_overall_score" in df.columns:
    valid = df.dropna(subset=["avg_overall_score"])
    if len(valid) > 0:
        plt.figure()
        plt.bar(valid["stage"], valid["avg_overall_score"])
        plt.title("Average Quality Score")
        plt.ylabel("Score")
        plt.ylim(0, 10)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig("avg_quality_score.png")
        plt.close()

print("Visualization saved:")
print("- num_papers_comparison.png")
print("- schema_completeness.png")
print("- abstract_coverage.png")
print("- avg_quality_score.png")

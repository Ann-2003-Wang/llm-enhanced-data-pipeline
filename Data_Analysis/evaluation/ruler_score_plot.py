import json
import numpy as np
import matplotlib.pyplot as plt


# ===== Utilities =====
def load_scores(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["summary"]


# ===== Load data =====
raw_scores = load_scores("quality_statistics_raw.json")
cleaned_scores = load_scores("quality_statistics_cleaned.json")
enhanced_scores = load_scores("quality_statistics_full_enhanced.json")

stages = ["Raw", "Cleaned", "Enhanced"]

metrics = [
    ("Completeness", "avg_completeness_score"),
    ("Recency", "avg_recency_score"),
    ("Content", "avg_content_quality"),
    ("Technical", "avg_technical_depth"),
    ("Overall", "overall_score"),
]

raw_values = [raw_scores[k] for _, k in metrics]
cleaned_values = [cleaned_scores[k] for _, k in metrics]
enhanced_values = [enhanced_scores[k] for _, k in metrics]


# ===== Plot =====
x = np.arange(len(metrics))
width = 0.25

plt.figure(figsize=(11, 5))

plt.bar(x - width, raw_values, width, label="Raw")
plt.bar(x, cleaned_values, width, label="Cleaned")
plt.bar(x + width, enhanced_values, width, label="Enhanced")

plt.xticks(x, [m[0] for m in metrics])
plt.ylabel("Average Score")
plt.title("Three-Stage Data Quality Improvement Comparison")
plt.legend()

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig("three_stage_quality_improvement_ruler_based.png", dpi=300)
plt.show()

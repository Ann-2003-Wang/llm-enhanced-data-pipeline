# data_quality_comparison.py

"""
Data Quality Comparison Script

This script analyzes and compares the quality of academic paper datasets across three processing stages:

1. Raw Merged Data (merged_papers.jsonl) - Unprocessed raw data
2. Cleaned & Aligned Data (papers_final_aligned.jsonl) - After cleaning and field alignment
3. Enhanced & Filtered Data (papers_master_final.jsonl) - After data augmentation and filtering

Key functionalities:
- Computes comprehensive quality metrics for each dataset including completeness percentages, average lengths, and schema compliance
- Generates a terminal display of quality comparison table
- Exports comparison results to CSV format (data_quality_comparison_3stage.csv)

The analysis helps track data quality improvements through the processing pipeline.
"""

import json
import pandas as pd


# Utils
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def safe_len(x):
    if isinstance(x, str):
        return len(x.strip())
    if isinstance(x, list):
        return len(x)
    return 0

# Statistics
def compute_stats(papers, stage_name):
    total = len(papers)

    stats = {
        "stage": stage_name,
        "num_papers": total,
        "has_abstract_%": 0,
        "has_authors_%": 0,
        "has_fields_%": 0,
        "has_keywords_%": 0,
        "avg_abstract_length": 0,
        "avg_title_length": 0,
        "schema_completeness_%": 0,
        "avg_overall_score": None,
    }

    if total == 0:
        return stats

    abstract_lens = []
    title_lens = []
    scores = []
    complete_schema = 0

    for p in papers:
        if safe_len(p.get("abstract")) > 0:
            stats["has_abstract_%"] += 1
            abstract_lens.append(len(p["abstract"]))

        if safe_len(p.get("authors")) > 0:
            stats["has_authors_%"] += 1

        if safe_len(p.get("fields_of_study")) > 0:
            stats["has_fields_%"] += 1

        if safe_len(p.get("keywords")) > 0:
            stats["has_keywords_%"] += 1

        if safe_len(p.get("title")) > 0:
            title_lens.append(len(p["title"]))

        # schema completeness
        required_fields = [
            "paper_id", "title", "abstract",
            "authors", "fields_of_study", "url"
        ]
        if all(p.get(f) for f in required_fields):
            complete_schema += 1

        # quality score (only exists after enhancement)
        qs = p.get("quality_scores")
        if isinstance(qs, dict) and "overall_score" in qs:
            scores.append(qs["overall_score"])

    # normalize
    for k in ["has_abstract_%", "has_authors_%", "has_fields_%", "has_keywords_%"]:
        stats[k] = round(stats[k] / total * 100, 2)

    stats["avg_abstract_length"] = round(
        sum(abstract_lens) / len(abstract_lens), 2
    ) if abstract_lens else 0

    stats["avg_title_length"] = round(
        sum(title_lens) / len(title_lens), 2
    ) if title_lens else 0

    stats["schema_completeness_%"] = round(
        complete_schema / total * 100, 2
    )

    if scores:
        stats["avg_overall_score"] = round(
            sum(scores) / len(scores), 2
        )

    return stats

# Main
if __name__ == "__main__":
    files = [
        ("merged_papers.jsonl", "Raw (Merged)"),
        ("papers_final_aligned.jsonl", "Cleaned & Aligned"),
        ("papers_master_final.jsonl", "Enhanced & Filtered"),
    ]

    rows = []
    for path, name in files:
        papers = load_jsonl(path)
        rows.append(compute_stats(papers, name))

    df = pd.DataFrame(rows)

    print("\n====== DATA QUALITY COMPARISON (3 STAGES) ======\n")
    print(df.to_string(index=False))

    df.to_csv("data_quality_comparison_3stage.csv", index=False)
    print("\nSaved to data_quality_comparison_3stage.csv")

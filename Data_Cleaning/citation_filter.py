# citation_filter.py

"""
Remove papers with abnormally low citation numbers (optional threshold) to ensure data quality.    
Since our project focuses on newly published papers, especially on timeliness, some of the relatively recent papers collected were not cited.    
Therefore, we have kept all of them. If there are specific requirements regarding the number of citations, we can make the necessary adjustments accordingly.
"""

import json

def load_jsonl(file_path):
    papers = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
    return papers

def save_jsonl(papers, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")

def filter_by_citations(papers, min_citations=0):
    filtered = [p for p in papers if p.get("citation_count", 0) >= min_citations]
    print(f"Citation filter: {len(papers)} -> {len(filtered)}")
    return filtered

if __name__ == "__main__":
    input_file = "papers_cleaned_text.jsonl"
    output_file = "papers_filtered_citations.jsonl"

    papers = load_jsonl(input_file)
    papers = filter_by_citations(papers, min_citations=0)
    save_jsonl(papers, output_file)

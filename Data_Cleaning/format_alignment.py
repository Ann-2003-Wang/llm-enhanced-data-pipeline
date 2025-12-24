# format_alignment.py
import json

REQUIRED_FIELDS = [
    "source", "paper_id", "title", "abstract", "abstract_source",
    "authors", "publish_year", "venue", "citation_count",
    "fields_of_study", "url"
]

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

def align_format(papers):
    aligned = []
    for paper in papers:
        new_paper = {}
        for field in REQUIRED_FIELDS:
            new_paper[field] = paper.get(field, "" if field != "authors" and field != "fields_of_study" else [])
        aligned.append(new_paper)
    return aligned

if __name__ == "__main__":
    input_file = "papers_cleaned_fields.jsonl"
    output_file = "papers_final_aligned.jsonl"

    papers = load_jsonl(input_file)
    papers = align_format(papers)
    save_jsonl(papers, output_file)
    print(f"Format alignment done. Papers count: {len(papers)}")

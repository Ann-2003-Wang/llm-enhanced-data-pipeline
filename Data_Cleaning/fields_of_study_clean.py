# fields_of_study_clean.py
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

def clean_fields_of_study(papers):
    for paper in papers:
        if 'fields_of_study' in paper and isinstance(paper['fields_of_study'], list):
            cleaned = list({f.strip().title() for f in paper['fields_of_study'] if f.strip()})
            paper['fields_of_study'] = cleaned
    return papers

if __name__ == "__main__":
    input_file = "papers_filtered_citations.jsonl"
    output_file = "papers_cleaned_fields.jsonl"

    papers = load_jsonl(input_file)
    papers = clean_fields_of_study(papers)
    save_jsonl(papers, output_file)

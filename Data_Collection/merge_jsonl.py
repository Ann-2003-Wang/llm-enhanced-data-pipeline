# merge_jsonl.py
import json

files = [
    "arxiv_papers.jsonl",
    "semantic_scholar_papers.jsonl",
    "pubmed_papers.jsonl"
]

merged_file = "merged_papers.jsonl"
seen_ids = set()
count = 0

with open(merged_file, "w", encoding="utf-8") as fout:
    for file in files:
        with open(file, "r", encoding="utf-8") as fin:
            for line in fin:
                data = json.loads(line.strip())
                pid = data.get("paper_id") or data.get("title")
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                    count += 1

print(f"Merged total {count} unique records into {merged_file}.")

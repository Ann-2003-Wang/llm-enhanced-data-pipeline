# strict_deduplication.py
import json
import hashlib
from collections import defaultdict

def load_jsonl(file_path):
    """Load JSONL file"""
    papers = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
    return papers

def save_jsonl(papers, file_path):
    """Save JSONL file"""
    with open(file_path, "w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")

# 1. Exact ID Deduplication
def dedup_by_id(papers):
    seen_ids = set()
    unique_papers = []
    for paper in papers:
        pid = paper.get("paper_id")
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            unique_papers.append(paper)
        elif not pid:
            unique_papers.append(paper)
    print(f"Step 1 - After ID deduplication: {len(unique_papers)} papers")
    return unique_papers

# 2. Title Hash Deduplication (exact match)
def dedup_by_title_hash(papers):
    seen_hashes = set()
    unique_papers = []
    for paper in papers:
        title = paper.get("title", "").strip().lower()
        h = hashlib.md5(title.encode("utf-8")).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_papers.append(paper)
    print(f"Step 2 - After title hash deduplication: {len(unique_papers)} papers")
    return unique_papers

# 3. Title Similarity Deduplication (intersection ratio threshold)
def dedup_by_title_similarity(papers, threshold=0.9):
    to_remove = set()
    n = len(papers)
    for i in range(n):
        if i in to_remove:
            continue
        title_i = papers[i].get("title", "").lower().split()
        set_i = set(title_i)
        if not set_i:
            continue
        for j in range(i+1, n):
            if j in to_remove:
                continue
            title_j = papers[j].get("title", "").lower().split()
            set_j = set(title_j)
            if not set_j:
                continue
            similarity = len(set_i & set_j) / len(set_i | set_j)
            if similarity >= threshold:
                # Keep the one with more recent publish_date
                date_i = papers[i].get("publish_year") or 0
                date_j = papers[j].get("publish_year") or 0
                if date_i >= date_j:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
    unique_papers = [papers[k] for k in range(n) if k not in to_remove]
    print(f"Step 3 - After title similarity deduplication: {len(unique_papers)} papers")
    return unique_papers


if __name__ == "__main__":
    input_file = "merged_papers.jsonl"
    output_file = "merged_papers_dedup.jsonl"

    papers = load_jsonl(input_file)
    print(f"Original paper count: {len(papers)}")

    # Perform deduplication in sequence
    papers = dedup_by_id(papers)
    papers = dedup_by_title_hash(papers)
    papers = dedup_by_title_similarity(papers, threshold=0.9)

    save_jsonl(papers, output_file)
    print(f"Final deduplicated paper count: {len(papers)}")
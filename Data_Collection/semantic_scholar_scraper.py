# semantic_scholar_scraper.py
import json
from semanticscholar import SemanticScholar

sch = SemanticScholar()  

def fetch_s2_papers(query, limit=200):
    print(f"Searching Semantic Scholar for '{query}' ...")
    results = sch.search_papers(query, limit=limit)
    papers = []

    for p in results:
        paper = {
            "source": "semantic_scholar",
            "paper_id": p.paperId,
            "title": p.title,
            "authors": [a.name for a in p.authors] if p.authors else [],
            "abstract": p.abstract or "",
            "publish_date": p.year,
            "url": p.externalUrls[0] if p.externalUrls else "",
            "fields_of_study": p.fieldsOfStudy
        }
        papers.append(paper)
    return papers

if __name__ == "__main__":
    topics = ["artificial intelligence", "machine learning", "computer vision"]
    all_papers = []

    for t in topics:
        s2papers = fetch_s2_papers(t, limit=200)
        all_papers.extend(s2papers)
        print(f"Got {len(s2papers)} from Semantic Scholar for '{t}'")

    with open("semantic_scholar_papers.jsonl", "w", encoding="utf-8") as f:
        for p in all_papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Saved total {len(all_papers)} papers from Semantic Scholar.")

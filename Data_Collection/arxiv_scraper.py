# arxiv_scraper.py
import arxiv
import json
from datetime import datetime, timedelta

def fetch_arxiv_papers(query, max_results=300, time_range="month"):
    end_date = datetime.now()
    if time_range == "week":
        start_date = end_date - timedelta(days=7)
    elif time_range == "months":
        start_date = end_date - timedelta(days=30)
    else:
        start_date = end_date - timedelta(days=30)

    time_filter = f"submittedDate:[{start_date.strftime('%Y%m%d')}+TO+{end_date.strftime('%Y%m%d')}]"
    full_query = f"{query}+AND+{time_filter}"

    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    results = []
    for r in search.results():
        paper = {
            "source": "arxiv",
            "paper_id": r.get_short_id(),
            "title": r.title,
            "authors": [str(a) for a in r.authors],
            "abstract": r.summary,
            "publish_date": str(r.published.date()),
            "url": f"http://arxiv.org/abs/{r.get_short_id()}",
            "categories": r.categories
        }
        results.append(paper)

    return results

if __name__ == "__main__":
    topics = [
    "artificial intelligence",
    "machine learning",
    "computer vision",
    "natural language processing",
    "robotics and human-computer interaction"
    ]

    all_papers = []

    for t in topics:
        papers = fetch_arxiv_papers(t, max_results=1000, time_range="month")
        all_papers.extend(papers)
        print(f"Fetched {len(papers)} from arXiv for '{t}'")

    with open("arxiv_papers.jsonl", "w", encoding="utf-8") as f:
        for p in all_papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Saved total {len(all_papers)} papers from arXiv.")

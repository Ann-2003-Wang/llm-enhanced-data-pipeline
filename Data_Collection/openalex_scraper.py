# openalex_scraper.py
import json
import time
import requests

BASE_URL = "https://api.openalex.org/works"


def fetch_openalex_papers(query, limit=200, per_page=50):

    print(f"Searching OpenAlex for '{query}' ...")

    papers = []
    page = 1
    collected = 0

    while collected < limit:
        params = {
            "search": query,
            "per-page": per_page,
            "page": page,
            "mailto": "research@example.com"  # OpenAlex 推荐填写
        }

        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            break

        for r in results:
            if collected >= limit:
                break

            paper = {
                "source": "openalex",
                "paper_id": r.get("id", ""),
                "title": r.get("title", ""),
                "abstract": r.get("abstract", "") or "",
                "authors": [
                    a["author"]["display_name"]
                    for a in r.get("authorships", [])
                    if "author" in a
                ],
                "publish_date": r.get("publication_year", ""),
                "venue": r.get("host_venue", {}).get("display_name", ""),
                "citation_count": r.get("cited_by_count", 0),
                "fields_of_study": [
                    c["display_name"]
                    for c in r.get("concepts", [])
                    if c.get("level", 10) <= 1
                ],
                "url": r.get("id", "")
            }

            papers.append(paper)
            collected += 1

        page += 1
        time.sleep(1) 

    return papers


if __name__ == "__main__":

    topics = [
        "artificial intelligence",
        "machine learning",
        "computer vision",
        "natural language processing",
        "robotics human computer interaction"
    ]

    all_papers = []

    for t in topics:
        papers = fetch_openalex_papers(t, limit=300)
        all_papers.extend(papers)
        print(f"Got {len(papers)} papers from OpenAlex for '{t}'")

    output_file = "openalex_papers.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for p in all_papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Saved total {len(all_papers)} papers to {output_file}")

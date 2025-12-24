# text_cleaning.py
import json
import re

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

def clean_text(papers):
    for paper in papers:
        # Clean title
        if 'title' in paper and paper['title'] is not None:
            # Normalize whitespace in title
            paper['title'] = re.sub(r'\s+', ' ', str(paper['title'])).strip()
        
        # Clean abstract
        if 'abstract' in paper and paper['abstract'] is not None:
            abstract = str(paper['abstract'])
            
            # Remove LaTeX inline math expressions: $...$
            abstract = re.sub(r'\$.*?\$', '', abstract)
            
            # Remove LaTeX display math expressions: $$...$$
            abstract = re.sub(r'\$\$.*?\$\$', '', abstract)
            
            # Remove LaTeX commands with braces: \command{...}
            abstract = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', abstract)
            
            # Remove standalone LaTeX commands
            abstract = re.sub(r'\\[a-zA-Z]+\s*', '', abstract)
            
            # Remove HTML entities
            abstract = re.sub(r'&[a-zA-Z]+;', '', abstract)
            
            # Remove non-ASCII characters (optional, can be adjusted)
            abstract = re.sub(r'[^\x00-\x7F]+', ' ', abstract)
            
            # Normalize whitespace
            abstract = re.sub(r'\s+', ' ', abstract).strip()
            
            paper['abstract'] = abstract
            paper['abstract_source'] = "original_cleaned"
        
        # Clean authors list
        if 'authors' in paper and isinstance(paper['authors'], list):
            cleaned_authors = []
            for author in paper['authors']:
                if author is not None:
                    # Convert to string and strip whitespace
                    cleaned_author = str(author).strip()
                    if cleaned_author:  # Only add non-empty strings
                        cleaned_authors.append(cleaned_author)
            paper['authors'] = cleaned_authors
    
    return papers

if __name__ == "__main__":
    # Input and output file paths
    input_file = "merged_papers_dedup.jsonl"
    output_file = "papers_cleaned_text.jsonl"
    
    papers = load_jsonl(input_file)
    
    papers = clean_text(papers)
    
    save_jsonl(papers, output_file)
    
    print(f"Text cleaning completed successfully.")
    print(f"Original count: {len(papers)} papers")
    print(f"Output saved to: {output_file}")
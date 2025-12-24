# build_simple_dataset.py

import json
from collections import defaultdict
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def safe_string(value: Any) -> str:
    """Safely convert value to string and clean."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()

def load_jsonl(path: str) -> Dict[str, Any]:
    """Load JSONL file into a dictionary with paper_id as key."""
    data = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    obj = json.loads(line)
                    # Try multiple possible ID fields
                    pid = obj.get("paper_id") or obj.get("id") or f"paper_{line_num}"
                    data[pid] = obj
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num} in {path}: JSON decode error - {e}")
                except Exception as e:
                    logger.warning(f"Line {line_num} in {path}: Error - {e}")
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
    
    logger.info(f"Loaded {len(data)} items from {path}")
    return data

def clean_list(lst: Any, max_len: int = None) -> List[str]:
    """Clean and deduplicate a list."""
    if not lst or not isinstance(lst, list):
        return []
    
    cleaned = []
    seen = set()
    
    for item in lst:
        if item is None:
            continue
        
        # Convert to string and clean
        if isinstance(item, str):
            clean_item = item.strip()
        else:
            clean_item = str(item).strip()
        
        if clean_item and clean_item not in seen:
            cleaned.append(clean_item)
            seen.add(clean_item)
            
        if max_len and len(cleaned) >= max_len:
            break
    
    return cleaned

def safe_int(value: Any) -> int:
    """Safely convert to integer."""
    if value is None:
        return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0

def safe_float(value: Any) -> float:
    """Safely convert to float."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def normalize_scores(scores: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize quality scores to ensure consistent structure."""
    if not scores or not isinstance(scores, dict):
        return {
            "novelty": 0,
            "technical_depth": 0,
            "clarity": 0,
            "impact_potential": 0,
            "overall_score": 0.0,
            "confidence": 0.0
        }
    
    normalized = {}
    
    # Integer scores (0-10)
    for key in ["novelty", "technical_depth", "clarity", "impact_potential"]:
        value = scores.get(key)
        if value is not None:
            int_val = safe_int(value)
            normalized[key] = max(0, min(10, int_val))  # Clamp to 0-10
        else:
            normalized[key] = 0
    
    # Float scores
    overall = safe_float(scores.get("overall_score"))
    normalized["overall_score"] = max(0.0, min(10.0, overall))  # Clamp to 0.0-10.0
    
    confidence = safe_float(scores.get("confidence"))
    normalized["confidence"] = max(0.0, min(1.0, confidence))  # Clamp to 0.0-1.0
    
    return normalized

def normalize_contributions(contrib: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize contribution summary to ensure consistent structure."""
    if not contrib or not isinstance(contrib, dict):
        return {
            "problem": "",
            "method": "",
            "key_contributions": [],
            "application_scenarios": []
        }
    
    normalized = {}
    
    # String fields
    normalized["problem"] = safe_string(contrib.get("problem"))
    normalized["method"] = safe_string(contrib.get("method"))
    
    # List fields
    normalized["key_contributions"] = clean_list(contrib.get("key_contributions"), max_len=10)
    normalized["application_scenarios"] = clean_list(contrib.get("application_scenarios"), max_len=10)
    
    return normalized

def main():
    """Simple merge without filtering."""
    logger.info("Starting simple merge of all papers...")
    
    # Load all data files
    base = load_jsonl("papers_final_aligned.jsonl")
    fields = load_jsonl("papers_enhanced_fields.jsonl")
    keywords = load_jsonl("papers_enhanced_keywords.jsonl")
    scores = load_jsonl("papers_enhanced_scores.jsonl")
    contribs = load_jsonl("papers_enhanced_contributions.jsonl")
    
    if not base:
        logger.error("No base papers found. Exiting.")
        return
    
    logger.info(f"Base papers: {len(base)}")
    logger.info(f"Fields data: {len(fields)}")
    logger.info(f"Keywords data: {len(keywords)}")
    logger.info(f"Scores data: {len(scores)}")
    logger.info(f"Contributions data: {len(contribs)}")
    
    final_papers = []
    stats = defaultdict(int)
    
    for pid, paper in base.items():
        merged = {}
        
        # Basic paper info (always include)
        merged["paper_id"] = pid
        merged["title"] = safe_string(paper.get("title"))
        merged["abstract"] = safe_string(paper.get("abstract"))
        merged["source"] = safe_string(paper.get("source"))
        merged["abstract_source"] = safe_string(paper.get("abstract_source"))
        
        # Optional fields (include if available)
        authors = paper.get("authors")
        if authors is not None:
            merged["authors"] = clean_list(authors, max_len=20)
            stats["has_authors"] += 1
        
        publish_year = paper.get("publish_year")
        if publish_year is not None:
            merged["publish_year"] = safe_int(publish_year)
            stats["has_publish_year"] += 1
        
        venue = paper.get("venue")
        if venue is not None:
            merged["venue"] = safe_string(venue)
            stats["has_venue"] += 1
        
        citation_count = paper.get("citation_count")
        if citation_count is not None:
            merged["citation_count"] = safe_int(citation_count)
            stats["has_citation_count"] += 1
        
        url = paper.get("url")
        if url is not None:
            merged["url"] = safe_string(url)
            stats["has_url"] += 1
        
        # Enhanced fields
        if pid in fields:
            merged["fields_of_study"] = clean_list(
                fields[pid].get("fields_of_study"), 
                max_len=8
            )
            stats["has_fields"] += 1
        else:
            merged["fields_of_study"] = []
            stats["missing_fields"] += 1
        
        # Keywords
        if pid in keywords:
            merged["keywords"] = clean_list(
                keywords[pid].get("keywords"), 
                max_len=8
            )
            stats["has_keywords"] += 1
        else:
            merged["keywords"] = []
            stats["missing_keywords"] += 1
        
        # Quality scores
        if pid in scores:
            scores_data = scores[pid].get("quality_scores", {})
            merged["quality_scores"] = normalize_scores(scores_data)
            stats["has_scores"] += 1
        else:
            merged["quality_scores"] = normalize_scores({})
            stats["missing_scores"] += 1
        
        # Contribution summary
        if pid in contribs:
            contrib_data = contribs[pid].get("contribution_summary", {})
            merged["contribution_summary"] = normalize_contributions(contrib_data)
            stats["has_contributions"] += 1
        else:
            merged["contribution_summary"] = normalize_contributions({})
            stats["missing_contributions"] += 1
        
        # Add to final papers (NO FILTERING)
        final_papers.append(merged)
    
    # Save all papers
    output_file = "papers_master_simple.jsonl"
    logger.info(f"Saving {len(final_papers)} papers to {output_file}")
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for paper in final_papers:
                f.write(json.dumps(paper, ensure_ascii=False) + "\n")
        
        logger.info("Successfully saved all papers")
    except Exception as e:
        logger.error(f"Failed to save output file: {e}")
        # Try backup
        backup_file = f"{output_file}.backup"
        try:
            with open(backup_file, "w", encoding="utf-8") as f:
                for paper in final_papers:
                    f.write(json.dumps(paper, ensure_ascii=False) + "\n")
            logger.info(f"Saved backup to {backup_file}")
        except Exception as e2:
            logger.error(f"Failed to save backup: {e2}")
    
    # Print statistics
    logger.info("=" * 60)
    logger.info("SIMPLE MERGE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total papers processed: {len(final_papers)}")
    logger.info(f"Total papers in base: {len(base)}")
    
    logger.info("-" * 60)
    logger.info("DATA AVAILABILITY STATISTICS:")
    logger.info(f"  Papers with authors: {stats['has_authors']} ({stats['has_authors']/len(base)*100:.1f}%)")
    logger.info(f"  Papers with publish_year: {stats['has_publish_year']} ({stats['has_publish_year']/len(base)*100:.1f}%)")
    logger.info(f"  Papers with venue: {stats['has_venue']} ({stats['has_venue']/len(base)*100:.1f}%)")
    logger.info(f"  Papers with citation_count: {stats['has_citation_count']} ({stats['has_citation_count']/len(base)*100:.1f}%)")
    logger.info(f"  Papers with url: {stats['has_url']} ({stats['has_url']/len(base)*100:.1f}%)")
    logger.info(f"  Papers with fields_of_study: {stats['has_fields']} ({stats['has_fields']/len(base)*100:.1f}%)")
    logger.info(f"  Papers with keywords: {stats['has_keywords']} ({stats['has_keywords']/len(base)*100:.1f}%)")
    logger.info(f"  Papers with quality_scores: {stats['has_scores']} ({stats['has_scores']/len(base)*100:.1f}%)")
    logger.info(f"  Papers with contribution_summary: {stats['has_contributions']} ({stats['has_contributions']/len(base)*100:.1f}%)")
    
    # Calculate average scores for all papers
    if final_papers:
        avg_scores = defaultdict(float)
        for paper in final_papers:
            qs = paper.get("quality_scores", {})
            for key in ["novelty", "technical_depth", "clarity", "impact_potential", "overall_score", "confidence"]:
                if key in qs:
                    avg_scores[key] += qs[key]
        
        logger.info("-" * 60)
        logger.info("AVERAGE QUALITY SCORES:")
        for key in ["novelty", "technical_depth", "clarity", "impact_potential"]:
            if key in avg_scores:
                avg = avg_scores[key] / len(final_papers)
                logger.info(f"  {key:20s}: {avg:.2f}/10")
        
        if "overall_score" in avg_scores:
            avg = avg_scores["overall_score"] / len(final_papers)
            logger.info(f"  overall_score:        {avg:.2f}/10")
        
        if "confidence" in avg_scores:
            avg = avg_scores["confidence"] / len(final_papers)
            logger.info(f"  confidence:           {avg:.3f}")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
# build_simple_dataset.py

# 
import json
from collections import defaultdict
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_jsonl(path):
    """Load JSONL file and return dictionary with paper_id as key."""
    data = {}
    skipped = 0
    
    if not Path(path).exists():
        logger.error(f"File not found: {path}")
        return data
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
                pid = obj.get("paper_id") or obj.get("id")
                
                if pid:
                    data[pid] = obj
                else:
                    logger.warning(f"Line {line_num}: No paper_id found")
                    skipped += 1
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error - {e}")
                skipped += 1
            except Exception as e:
                logger.warning(f"Line {line_num}: Unexpected error - {e}")
                skipped += 1
    
    logger.info(f"Loaded {len(data)} papers from {path}, skipped {skipped} lines")
    return data

def clean_string(value):
    """Safely clean string, handling None values."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()

def clean_list(lst, max_len=None):
    """Clean a list, handling None values and duplicates."""
    if not lst or not isinstance(lst, list):
        return []
    
    cleaned = []
    seen = set()
    
    for x in lst:
        if x is None:
            continue
            
        if isinstance(x, str):
            item = x.strip()
        else:
            item = str(x).strip()
            
        if item and item not in seen:
            cleaned.append(item)
            seen.add(item)
            
        if max_len and len(cleaned) >= max_len:
            break
    
    return cleaned

def safe_int(value):
    """Safely convert to integer."""
    if value is None:
        return None
    
    try:
        # Handle string, int, float
        if isinstance(value, str):
            # Remove non-numeric characters from end
            cleaned = value.strip()
            # Try to extract first number
            import re
            match = re.search(r'[-+]?\d+', cleaned)
            if match:
                return int(match.group())
            return None
        elif isinstance(value, (int, float)):
            return int(value)
        else:
            return None
    except (ValueError, TypeError):
        return None

def safe_float(value):
    """Safely convert to float."""
    if value is None:
        return None
    
    try:
        if isinstance(value, str):
            cleaned = value.strip()
            # Try to parse as float
            try:
                return float(cleaned)
            except ValueError:
                # Try to extract number
                import re
                match = re.search(r'[-+]?\d*\.?\d+', cleaned)
                if match:
                    return float(match.group())
                return None
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return None
    except (ValueError, TypeError):
        return None

def get_nested_value(obj, keys, default=None):
    """Safely get nested value from dictionary."""
    if not obj or not isinstance(obj, dict):
        return default
    
    current = obj
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    
    return current

def main():
    # Define file paths
    base_path = "papers_final_aligned.jsonl"
    fields_path = "papers_enhanced_fields.jsonl"
    keywords_path = "papers_enhanced_keywords.jsonl"
    scores_path = "papers_enhanced_scores.jsonl"
    contribs_path = "papers_enhanced_contributions.jsonl"
    output_path = "papers_master_final.jsonl"
    
    logger.info("Loading data files...")
    
    # Load all data files
    base = load_jsonl(base_path)
    fields = load_jsonl(fields_path)
    keywords = load_jsonl(keywords_path)
    scores = load_jsonl(scores_path)
    contribs = load_jsonl(contribs_path)
    
    if not base:
        logger.error("No base papers loaded. Exiting.")
        return
    
    logger.info(f"Base papers: {len(base)}")
    logger.info(f"Fields data: {len(fields)}")
    logger.info(f"Keywords data: {len(keywords)}")
    logger.info(f"Scores data: {len(scores)}")
    logger.info(f"Contributions data: {len(contribs)}")
    
    final_papers = []
    dropped = 0
    merge_stats = defaultdict(int)
    
    logger.info("Merging and filtering papers...")
    
    for pid, paper in base.items():
        merged = {}
        
        # Basic paper info
        merged["source"] = paper.get("source")
        merged["paper_id"] = pid
        merged["title"] = clean_string(paper.get("title"))
        merged["abstract"] = clean_string(paper.get("abstract"))
        merged["abstract_source"] = paper.get("abstract_source")
        merged["authors"] = clean_list(paper.get("authors"), max_len=20)
        merged["publish_year"] = safe_int(paper.get("publish_year"))
        merged["venue"] = clean_string(paper.get("venue"))
        merged["citation_count"] = safe_int(paper.get("citation_count")) or 0
        merged["url"] = paper.get("url")
        
        # Fields of study
        if pid in fields:
            merged["fields_of_study"] = clean_list(
                get_nested_value(fields[pid], ["fields_of_study"]), 
                max_len=8
            )
            merge_stats["has_fields"] += 1
        else:
            merged["fields_of_study"] = []
            merge_stats["missing_fields"] += 1
        
        # Keywords
        if pid in keywords:
            merged["keywords"] = clean_list(
                get_nested_value(keywords[pid], ["keywords"]), 
                max_len=8
            )
            merge_stats["has_keywords"] += 1
        else:
            merged["keywords"] = []
            merge_stats["missing_keywords"] += 1
        
        # Quality scores
        if pid in scores:
            scores_data = get_nested_value(scores[pid], ["quality_scores"], {})
            # Ensure all score fields exist and are valid
            quality_scores = {}
            
            # Integer scores (0-10)
            for int_key in ["novelty", "technical_depth", "clarity", "impact_potential"]:
                value = scores_data.get(int_key)
                int_val = safe_int(value)
                if int_val is not None and 0 <= int_val <= 10:
                    quality_scores[int_key] = int_val
                else:
                    quality_scores[int_key] = 0
            
            # Float scores
            overall = safe_float(scores_data.get("overall_score"))
            if overall is not None and 0 <= overall <= 10:
                quality_scores["overall_score"] = round(overall, 1)
            else:
                quality_scores["overall_score"] = 0.0
            
            confidence = safe_float(scores_data.get("confidence"))
            if confidence is not None and 0 <= confidence <= 1:
                quality_scores["confidence"] = round(confidence, 2)
            else:
                quality_scores["confidence"] = 0.0
            
            merged["quality_scores"] = quality_scores
            merge_stats["has_scores"] += 1
        else:
            merged["quality_scores"] = {
                "novelty": 0,
                "technical_depth": 0,
                "clarity": 0,
                "impact_potential": 0,
                "overall_score": 0.0,
                "confidence": 0.0
            }
            merge_stats["missing_scores"] += 1
        
        # Contribution summary
        if pid in contribs:
            contrib_data = get_nested_value(contribs[pid], ["contribution_summary"], {})
            # Clean and validate contribution summary
            cleaned_contrib = {}
            
            # Problem and method (strings)
            for str_key in ["problem", "method"]:
                value = contrib_data.get(str_key)
                if isinstance(value, str) and value.strip():
                    cleaned_contrib[str_key] = value.strip()[:500]  # Limit length
                else:
                    cleaned_contrib[str_key] = ""
            
            # Lists
            for list_key in ["key_contributions", "application_scenarios"]:
                value = contrib_data.get(list_key)
                if isinstance(value, list):
                    cleaned_contrib[list_key] = clean_list(value, max_len=10)
                else:
                    cleaned_contrib[list_key] = []
            
            merged["contribution_summary"] = cleaned_contrib
            merge_stats["has_contribs"] += 1
        else:
            merged["contribution_summary"] = {
                "problem": "",
                "method": "",
                "key_contributions": [],
                "application_scenarios": []
            }
            merge_stats["missing_contribs"] += 1
        
        # Apply quality filters
        title = merged["title"]
        abstract = merged["abstract"]
        qs = merged["quality_scores"]
        
        try:
            # Check if paper meets minimum quality criteria
            title_too_short = len(title) < 8
            abstract_too_short = len(abstract) < 120
            score_too_low = qs.get("overall_score", 0) < 6.5
            depth_too_low = qs.get("technical_depth", 0) < 6
            confidence_too_low = qs.get("confidence", 0) < 0.6
            
            if (title_too_short or abstract_too_short or 
                score_too_low or depth_too_low or confidence_too_low):
                dropped += 1
                
                # Log reason for dropping (for debugging)
                if title_too_short:
                    merge_stats["dropped_title"] += 1
                if abstract_too_short:
                    merge_stats["dropped_abstract"] += 1
                if score_too_low:
                    merge_stats["dropped_score"] += 1
                if depth_too_low:
                    merge_stats["dropped_depth"] += 1
                if confidence_too_low:
                    merge_stats["dropped_confidence"] += 1
                    
                continue
            
            # Additional validation
            if not title or not abstract:
                dropped += 1
                merge_stats["dropped_empty"] += 1
                continue
            
            final_papers.append(merged)
            
        except Exception as e:
            logger.warning(f"Error filtering paper {pid}: {e}")
            dropped += 1
            merge_stats["dropped_error"] += 1
            continue
    
    # Save final papers
    logger.info(f"Saving {len(final_papers)} papers to {output_path}")
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for p in final_papers:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        
        logger.info("Successfully saved final papers")
    except Exception as e:
        logger.error(f"Failed to save output file: {e}")
        # Try to save backup
        backup_path = f"{output_path}.backup"
        try:
            with open(backup_path, "w", encoding="utf-8") as f:
                for p in final_papers:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            logger.info(f"Saved backup to {backup_path}")
        except:
            logger.error("Failed to save backup")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("MASTER MERGE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Base papers: {len(base)}")
    logger.info(f"Final papers: {len(final_papers)}")
    logger.info(f"Dropped papers: {dropped}")
    logger.info(f"Retention rate: {len(final_papers)/len(base)*100:.1f}%")
    
    logger.info("-" * 60)
    logger.info("DATA AVAILABILITY:")
    logger.info(f"Papers with fields: {merge_stats['has_fields']} ({merge_stats['has_fields']/len(base)*100:.1f}%)")
    logger.info(f"Papers with keywords: {merge_stats['has_keywords']} ({merge_stats['has_keywords']/len(base)*100:.1f}%)")
    logger.info(f"Papers with scores: {merge_stats['has_scores']} ({merge_stats['has_scores']/len(base)*100:.1f}%)")
    logger.info(f"Papers with contributions: {merge_stats['has_contribs']} ({merge_stats['has_contribs']/len(base)*100:.1f}%)")
    
    if dropped > 0:
        logger.info("-" * 60)
        logger.info("DROP REASONS:")
        if merge_stats["dropped_title"]:
            logger.info(f"  Title too short: {merge_stats['dropped_title']}")
        if merge_stats["dropped_abstract"]:
            logger.info(f"  Abstract too short: {merge_stats['dropped_abstract']}")
        if merge_stats["dropped_score"]:
            logger.info(f"  Score too low: {merge_stats['dropped_score']}")
        if merge_stats["dropped_depth"]:
            logger.info(f"  Technical depth too low: {merge_stats['dropped_depth']}")
        if merge_stats["dropped_confidence"]:
            logger.info(f"  Confidence too low: {merge_stats['dropped_confidence']}")
        if merge_stats["dropped_empty"]:
            logger.info(f"  Empty title/abstract: {merge_stats['dropped_empty']}")
        if merge_stats["dropped_error"]:
            logger.info(f"  Processing error: {merge_stats['dropped_error']}")
    
    # Calculate average scores for final papers
    if final_papers:
        avg_scores = defaultdict(float)
        for paper in final_papers:
            qs = paper.get("quality_scores", {})
            for key in ["novelty", "technical_depth", "clarity", "impact_potential", "overall_score", "confidence"]:
                if key in qs:
                    avg_scores[key] += qs[key]
        
        logger.info("-" * 60)
        logger.info("AVERAGE SCORES IN FINAL DATASET:")
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
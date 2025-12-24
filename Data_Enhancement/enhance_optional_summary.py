# enhance_optional_summary.py

"""
Enhanced paper contribution summarization using DeepSeek API with concurrent processing.

This script processes academic papers from a JSONL file, calls the DeepSeek API
to extract structured research contributions for each paper, and saves the enhanced data
to a new JSONL file with progress tracking, error handling, and checkpointing.
"""

import json
import requests
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random
from datetime import datetime
import backoff
from threading import Lock
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('contribution_extraction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API Configuration
API_KEY = "sk-79990d599cd74bc0a56f6ca2f200a621"
API_URL = "https://api.deepseek.com/v1/chat/completions"
REQUEST_TIMEOUT = 45  # Increased for summarization task
MAX_WORKERS = 5  # Conservative for summarization
BASE_DELAY = 0.7  # Base delay between requests
MAX_RETRIES = 5
BATCH_SIZE = 25  # Save progress every N papers
MAX_TOKENS = 400  # Increased for longer summaries

# Rate limiting and tracking
rate_limit_lock = Lock()
last_request_time = 0
request_counter = 0
error_counter = 0

# Expected contribution summary structure
CONTRIBUTION_STRUCTURE = {
    "problem": str,
    "method": str,
    "key_contributions": list,
    "application_scenarios": list
}

def rate_limited_request():
    """Ensure minimum delay between API requests with jitter."""
    global last_request_time
    
    with rate_limit_lock:
        current_time = time.time()
        elapsed = current_time - last_request_time
        if elapsed < BASE_DELAY:
            sleep_time = BASE_DELAY - elapsed + random.uniform(0, 0.15)
            time.sleep(sleep_time)
        last_request_time = time.time()

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, KeyError, json.JSONDecodeError),
    max_tries=MAX_RETRIES,
    max_time=300,
    jitter=backoff.full_jitter
)
def call_deepseek_with_backoff(prompt: str, paper_id: str = "unknown") -> Optional[str]:
    """
    Call DeepSeek API with exponential backoff for contribution extraction.
    
    Args:
        prompt: The prompt to send to the API
        paper_id: Paper identifier for logging
    
    Returns:
        API response content as string, or None if all retries fail
    """
    global request_counter, error_counter
    
    rate_limited_request()
    request_counter += 1
    
    try:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.25,
                "max_tokens": MAX_TOKENS
            },
            timeout=REQUEST_TIMEOUT
        )
        
        # Handle rate limiting
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 75))
            logger.warning(f"Rate limited for paper {paper_id}. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            response.raise_for_status()
        
        response.raise_for_status()
        
        data = response.json()
        
        # Validate response structure
        if not isinstance(data, dict):
            raise ValueError(f"Response is not a dictionary: {type(data)}")
        
        choices = data.get("choices", [])
        if not choices:
            raise ValueError(f"No choices in response: {data}")
        
        message = choices[0].get("message", {})
        content = message.get("content", "").strip()
        
        if not content:
            raise ValueError("Empty response content")
        
        # Log progress periodically
        if request_counter % 40 == 0:
            logger.info(f"Completed {request_counter} summarization requests")
        
        return content
        
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout for paper {paper_id}")
        raise
    except requests.exceptions.HTTPError as e:
        error_counter += 1
        status_code = e.response.status_code if e.response else "unknown"
        logger.error(f"HTTP error {status_code} for paper {paper_id}")
        if status_code >= 500:
            time.sleep(12)  # Longer sleep for server errors
        raise
    except Exception as e:
        error_counter += 1
        logger.error(f"API error for paper {paper_id}: {str(e)[:100]}")
        raise

def call_deepseek(prompt: str, paper_id: str = "unknown") -> Optional[str]:
    """
    Wrapper for API call with fallback handling.
    
    Args:
        prompt: The prompt to send to the API
        paper_id: Paper identifier for logging
    
    Returns:
        API response content as string, or None if all retries fail
    """
    try:
        return call_deepseek_with_backoff(prompt, paper_id)
    except Exception as e:
        logger.error(f"All retries failed for paper {paper_id}: {str(e)[:100]}")
        return None

def validate_contribution_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean the contribution summary structure.
    
    Args:
        summary: Raw summary dictionary
    
    Returns:
        Validated and cleaned summary dictionary
    """
    if not isinstance(summary, dict):
        return {}
    
    validated = {}
    
    # Validate each expected field
    for field, expected_type in CONTRIBUTION_STRUCTURE.items():
        if field in summary:
            value = summary[field]
            
            # Convert to appropriate type
            if expected_type == str:
                if isinstance(value, str):
                    validated[field] = value.strip()
                elif value is not None:
                    validated[field] = str(value).strip()
                else:
                    validated[field] = ""
            
            elif expected_type == list:
                if isinstance(value, list):
                    # Clean list items
                    cleaned_list = []
                    for item in value:
                        if isinstance(item, str):
                            cleaned = item.strip()
                            if cleaned:
                                cleaned_list.append(cleaned)
                        elif item is not None:
                            cleaned = str(item).strip()
                            if cleaned:
                                cleaned_list.append(cleaned)
                    validated[field] = cleaned_list
                else:
                    validated[field] = []
        
        else:
            # Set default values for missing fields
            if expected_type == str:
                validated[field] = ""
            else:
                validated[field] = []
    
    # Additional validation
    if validated.get("problem") and len(validated["problem"]) > 500:
        validated["problem"] = validated["problem"][:500] + "..."
    
    if validated.get("method") and len(validated["method"]) > 500:
        validated["method"] = validated["method"][:500] + "..."
    
    # Limit list lengths
    for list_field in ["key_contributions", "application_scenarios"]:
        if list_field in validated:
            validated[list_field] = validated[list_field][:10]  # Max 10 items
    
    return validated

def extract_contribution_summary(paper: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured contribution summary from paper metadata using DeepSeek API.
    
    Args:
        paper: Dictionary containing paper metadata
    
    Returns:
        Dictionary with original paper data plus contribution summary
    """
    paper_id = paper.get("paper_id", paper.get("id", "unknown"))
    
    # Prepare content
    title = paper.get("title", "").strip()
    abstract = paper.get("abstract", "").strip()
    
    # Truncate if too long
    if len(abstract) > 3500:
        abstract = abstract[:3500] + "..."
    
    prompt = f"""
You are an expert research analyst specializing in academic paper analysis.

Summarize the paper into structured research contributions based on the title and abstract.

IMPORTANT: Return ONLY a valid JSON object with the following keys:
- "problem": A concise description of the research problem addressed (1-2 sentences)
- "method": A brief overview of the methodology or approach used (1-2 sentences)
- "key_contributions": A list of 3-5 key contributions or innovations
- "application_scenarios": A list of 2-4 potential application scenarios

Guidelines:
- Be specific and technical, not generic
- Focus on what is novel or significant
- For contributions: use bullet-point style but return as JSON list
- For application scenarios: be concrete about where this research could be applied
- Keep each contribution and scenario item concise (1 sentence each)

Title:
{title}

Abstract:
{abstract}
"""
    summary = {}
    
    try:
        api_response = call_deepseek(prompt, paper_id)
        
        if api_response is None:
            logger.debug(f"No API response for paper {paper_id}")
        else:
            # Clean response - remove markdown code blocks and whitespace
            clean_response = api_response.strip()
            clean_response = re.sub(r'^```(?:json)?\s*', '', clean_response, flags=re.IGNORECASE)
            clean_response = re.sub(r'\s*```$', '', clean_response)
            
            # Try to parse as JSON
            try:
                parsed = json.loads(clean_response)
                summary = validate_contribution_summary(parsed)
                
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON for paper {paper_id}: {str(e)[:100]}")
                
                # Advanced JSON extraction
                json_patterns = [
                    r'\{[^{}]*\}',  # Simple object
                    r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested object
                ]
                
                for pattern in json_patterns:
                    matches = re.finditer(pattern, clean_response, re.DOTALL)
                    for match in matches:
                        try:
                            parsed = json.loads(match.group(0))
                            if isinstance(parsed, dict):
                                temp_summary = validate_contribution_summary(parsed)
                                # Check if we have at least some valid structure
                                if (temp_summary.get("problem") or 
                                    temp_summary.get("method") or 
                                    temp_summary.get("key_contributions")):
                                    summary = temp_summary
                                    break
                        except:
                            continue
                    if summary:
                        break
                
                # Fallback: extract structured information from text
                if not summary:
                    logger.debug(f"Using advanced text extraction for paper {paper_id}")
                    summary = extract_summary_from_text(clean_response)
    
    except Exception as e:
        logger.error(f"Unexpected error summarizing paper {paper_id}: {str(e)[:100]}")
    
    # Final validation
    summary = validate_contribution_summary(summary)
    
    # Create enhanced paper
    enhanced_paper = paper.copy()
    enhanced_paper["contribution_summary"] = summary
    enhanced_paper["summary_extraction_time"] = datetime.now().isoformat()
    
    return enhanced_paper

def extract_summary_from_text(text: str) -> Dict[str, Any]:
    """
    Extract structured summary from text response when JSON parsing fails.
    
    Args:
        text: Raw text response from API
    
    Returns:
        Structured summary dictionary
    """
    summary = {
        "problem": "",
        "method": "",
        "key_contributions": [],
        "application_scenarios": []
    }
    
    # Extract problem and method using keyword patterns
    lines = text.split('\n')
    
    # Look for problem description
    problem_patterns = [
        r'[Pp]roblem[:：]?\s*(.+)',
        r'[Rr]esearch\s+[Pp]roblem[:：]?\s*(.+)',
        r'[Tt]he\s+problem\s+is\s+(.+)',
    ]
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Check for problem
        for pattern in problem_patterns:
            match = re.search(pattern, line)
            if match:
                summary["problem"] = match.group(1).strip()
                break
        
        # Check for method
        method_patterns = [
            r'[Mm]ethod(?:ology)?[:：]?\s*(.+)',
            r'[Aa]pproach[:：]?\s*(.+)',
            r'[Tt]echnique[:：]?\s*(.+)',
        ]
        
        for pattern in method_patterns:
            match = re.search(pattern, line)
            if match:
                summary["method"] = match.group(1).strip()
                break
        
        # Look for contributions (bullet points, numbered lists, or after headings)
        if re.match(r'^[Cc]ontributions?[:：]?', line) or re.match(r'^[Kk]ey\s+[Cc]ontributions?[:：]?', line):
            # Collect next lines as contributions
            for j in range(i + 1, min(i + 10, len(lines))):
                next_line = lines[j].strip()
                if next_line and not re.match(r'^[A-Za-z\s]+[:：]?$', next_line):
                    # Clean bullet points or numbers
                    clean_line = re.sub(r'^[•\-\*\d\.\s]+', '', next_line)
                    if clean_line and len(clean_line) > 5:
                        summary["key_contributions"].append(clean_line)
            
            # Limit to 5 contributions
            summary["key_contributions"] = summary["key_contributions"][:5]
        
        # Look for applications
        if re.match(r'^[Aa]pplications?[:：]?', line) or re.match(r'^[Uu]ses?[:：]?', line):
            # Collect next lines as applications
            for j in range(i + 1, min(i + 8, len(lines))):
                next_line = lines[j].strip()
                if next_line and not re.match(r'^[A-Za-z\s]+[:：]?$', next_line):
                    clean_line = re.sub(r'^[•\-\*\d\.\s]+', '', next_line)
                    if clean_line and len(clean_line) > 5:
                        summary["application_scenarios"].append(clean_line)
            
            # Limit to 4 applications
            summary["application_scenarios"] = summary["application_scenarios"][:4]
    
    return summary

def load_papers(input_path: Path) -> List[Dict[str, Any]]:
    """Load papers from JSONL file."""
    papers = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    papers.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
        
        logger.info(f"Loaded {len(papers)} papers from {input_path}")
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        raise
    
    return papers

def save_checkpoint(papers: List[Dict[str, Any]], checkpoint_path: Path):
    """Save intermediate results to checkpoint file."""
    try:
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            for paper in papers:
                f.write(json.dumps(paper, ensure_ascii=False) + '\n')
        
        logger.debug(f"Checkpoint saved: {len(papers)} papers")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

def load_checkpoint(checkpoint_path: Path) -> Tuple[List[Dict[str, Any]], set]:
    """Load papers from checkpoint file and get processed paper IDs."""
    if not checkpoint_path.exists():
        return [], set()
    
    papers = []
    processed_ids = set()
    
    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    paper = json.loads(line.strip())
                    papers.append(paper)
                    paper_id = paper.get("paper_id", paper.get("id"))
                    if paper_id:
                        processed_ids.add(paper_id)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(papers)} papers from checkpoint")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
    
    return papers, processed_ids

def process_concurrently(papers: List[Dict[str, Any]], 
                        max_workers: int = MAX_WORKERS,
                        checkpoint_path: Optional[Path] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Process papers concurrently using ThreadPoolExecutor.
    
    Args:
        papers: List of papers to process
        max_workers: Maximum concurrent threads
        checkpoint_path: Optional checkpoint file path
    
    Returns:
        Tuple of (enhanced papers, statistics)
    """
    enhanced_papers = []
    stats = {
        "total": len(papers),
        "successful": 0,
        "failed": 0,
        "complete_summaries": 0,
        "partial_summaries": 0,
        "empty_summaries": 0
    }
    
    # Adjust workers for summarization task
    effective_workers = min(max_workers, 6)
    logger.info(f"Summarizing {len(papers)} papers with {effective_workers} workers")
    
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        # Submit tasks
        future_to_index = {
            executor.submit(extract_contribution_summary, paper): i 
            for i, paper in enumerate(papers)
        }
        
        # Track completion order for proper ordering
        results = [None] * len(papers)
        
        # Process with progress bar
        with tqdm(total=len(papers), desc="Extracting contributions", unit="paper") as pbar:
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                
                try:
                    enhanced_paper = future.result(timeout=REQUEST_TIMEOUT + 50)
                    results[idx] = enhanced_paper
                    
                    summary = enhanced_paper.get("contribution_summary", {})
                    
                    if summary:
                        # Check completeness
                        has_problem = bool(summary.get("problem"))
                        has_method = bool(summary.get("method"))
                        has_contributions = bool(summary.get("key_contributions"))
                        has_applications = bool(summary.get("application_scenarios"))
                        
                        complete_fields = sum([has_problem, has_method, has_contributions, has_applications])
                        
                        if complete_fields >= 3:
                            stats["complete_summaries"] += 1
                            stats["successful"] += 1
                        elif complete_fields >= 1:
                            stats["partial_summaries"] += 1
                            stats["successful"] += 1
                        else:
                            stats["empty_summaries"] += 1
                            stats["failed"] += 1
                    else:
                        stats["empty_summaries"] += 1
                        stats["failed"] += 1
                        
                except Exception as e:
                    logger.error(f"Processing failed for paper at index {idx}: {str(e)[:100]}")
                    stats["failed"] += 1
                    # Add fallback
                    fallback_paper = papers[idx].copy()
                    fallback_paper["contribution_summary"] = {}
                    results[idx] = fallback_paper
                
                pbar.update(1)
                
                # Periodic checkpoint
                if checkpoint_path and pbar.n % BATCH_SIZE == 0:
                    # Collect completed results
                    completed = [r for r in results if r is not None]
                    save_checkpoint(completed, checkpoint_path)
                    
                    # Update progress in log
                    progress_pct = pbar.n / len(papers) * 100
                    logger.info(f"Progress: {pbar.n}/{len(papers)} papers summarized "
                               f"({progress_pct:.1f}%)")
    
    # Filter out None results
    enhanced_papers = [r for r in results if r is not None]
    
    return enhanced_papers, stats

def calculate_summary_statistics(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics on the collected summaries."""
    summaries = [p.get("contribution_summary", {}) for p in papers]
    
    stats = {
        "total_papers": len(papers),
        "papers_with_summary": 0,
        "field_completeness": {},
        "average_lengths": {},
        "summary_quality": {}
    }
    
    valid_summaries = [s for s in summaries if s]
    stats["papers_with_summary"] = len(valid_summaries)
    
    if not valid_summaries:
        return stats
    
    # Field completeness
    for field in CONTRIBUTION_STRUCTURE.keys():
        has_field = sum(1 for s in valid_summaries if s.get(field))
        stats["field_completeness"][field] = {
            "count": has_field,
            "percentage": has_field / len(valid_summaries) * 100
        }
    
    # Average lengths
    for field in ["problem", "method"]:
        lengths = [len(s.get(field, "")) for s in valid_summaries if s.get(field)]
        if lengths:
            stats["average_lengths"][field] = {
                "mean": sum(lengths) / len(lengths),
                "min": min(lengths),
                "max": max(lengths)
            }
    
    for list_field in ["key_contributions", "application_scenarios"]:
        counts = [len(s.get(list_field, [])) for s in valid_summaries]
        if counts:
            stats["average_lengths"][list_field] = {
                "mean": sum(counts) / len(counts),
                "min": min(counts),
                "max": max(counts)
            }
    
    # Summary quality scoring
    quality_scores = []
    for summary in valid_summaries:
        score = 0
        if summary.get("problem"):
            score += 1
        if summary.get("method"):
            score += 1
        if summary.get("key_contributions"):
            score += 1
        if summary.get("application_scenarios"):
            score += 1
        quality_scores.append(score)
    
    if quality_scores:
        stats["summary_quality"] = {
            "average_score": sum(quality_scores) / len(quality_scores),
            "distribution": {
                "0_fields": quality_scores.count(0),
                "1_field": quality_scores.count(1),
                "2_fields": quality_scores.count(2),
                "3_fields": quality_scores.count(3),
                "4_fields": quality_scores.count(4)
            }
        }
    
    return stats

def main():
    """Main execution function."""
    
    # File paths
    input_file = Path("papers_final_aligned.jsonl")
    output_file = Path("papers_enhanced_contributions.jsonl")
    checkpoint_file = Path("contribution_summary_checkpoint.jsonl")
    stats_file = Path("contribution_statistics.json")
    
    start_time = time.time()
    
    try:
        # Load papers
        papers = load_papers(input_file)
        
        if not papers:
            logger.error("No papers loaded. Exiting.")
            return
        
        # Load checkpoint
        checkpoint_papers, processed_ids = load_checkpoint(checkpoint_file)
        
        # Determine which papers need processing
        if processed_ids:
            papers_to_process = []
            for paper in papers:
                paper_id = paper.get("paper_id", paper.get("id"))
                if paper_id not in processed_ids:
                    papers_to_process.append(paper)
            
            logger.info(f"Resuming: {len(checkpoint_papers)} already processed, "
                       f"{len(papers_to_process)} remaining")
            
            # Process remaining papers
            enhanced_new_papers, stats = process_concurrently(
                papers_to_process,
                checkpoint_path=checkpoint_file
            )
            
            # Combine results
            enhanced_papers = checkpoint_papers + enhanced_new_papers
        else:
            # Process all papers
            enhanced_papers, stats = process_concurrently(
                papers,
                checkpoint_path=checkpoint_file
            )
        
        # Save final results
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for paper in enhanced_papers:
                    f.write(json.dumps(paper, ensure_ascii=False) + '\n')
            logger.info(f"Saved {len(enhanced_papers)} papers to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save output: {e}")
            # Save to backup file
            backup_file = Path(f"{output_file.stem}_backup_{int(time.time())}.jsonl")
            with open(backup_file, 'w', encoding='utf-8') as f:
                for paper in enhanced_papers:
                    f.write(json.dumps(paper, ensure_ascii=False) + '\n')
            logger.info(f"Saved backup to {backup_file}")
        
        # Calculate and save statistics
        summary_stats = calculate_summary_statistics(enhanced_papers)
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(summary_stats, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved contribution statistics to {stats_file}")
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")
        
        # Print summary
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info("=" * 70)
        logger.info("CONTRIBUTION SUMMARIZATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total papers: {stats['total']}")
        logger.info(f"Successfully processed: {stats['successful']}")
        logger.info(f"Complete summaries (3+ fields): {stats['complete_summaries']}")
        logger.info(f"Partial summaries (1-2 fields): {stats['partial_summaries']}")
        logger.info(f"Empty summaries: {stats['empty_summaries']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Total API requests: {request_counter}")
        logger.info(f"API errors: {error_counter}")
        if request_counter > 0:
            success_rate = (request_counter - error_counter) / request_counter * 100
            logger.info(f"API success rate: {success_rate:.1f}%")
        logger.info(f"Total time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        logger.info(f"Processing rate: {stats['total'] / elapsed_time:.2f} papers/sec")
        
        # Show summary statistics if available
        if summary_stats["papers_with_summary"] > 0:
            logger.info("-" * 70)
            logger.info("SUMMARY STATISTICS:")
            logger.info(f"Papers with summary: {summary_stats['papers_with_summary']} "
                       f"({summary_stats['papers_with_summary']/summary_stats['total_papers']*100:.1f}%)")
            
            for field, completeness in summary_stats["field_completeness"].items():
                logger.info(f"  {field:20s}: {completeness['count']:4d} papers "
                           f"({completeness['percentage']:.1f}%)")
            
            if summary_stats["summary_quality"]:
                quality = summary_stats["summary_quality"]
                logger.info(f"  Average quality score: {quality['average_score']:.1f}/4.0")
        
        logger.info("=" * 70)
        
        # Clean up checkpoint
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                logger.info(f"Removed checkpoint file: {checkpoint_file}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint: {e}")
        
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user.")
        logger.info(f"Checkpoint saved. Resume by running the script again.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
# enhance_scoring.py

"""
Enhanced academic paper scoring using DeepSeek API with concurrent processing.

This script processes academic papers from a JSONL file, calls the DeepSeek API
to evaluate paper quality across multiple dimensions, and saves the enhanced data
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
        logging.FileHandler('paper_scoring.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API Configuration
API_KEY = "sk-79990d599cd74bc0a56f6ca2f200a621"
API_URL = "https://api.deepseek.com/v1/chat/completions"
REQUEST_TIMEOUT = 40  # Increased for scoring task
MAX_WORKERS = 6  # Conservative for scoring task
BASE_DELAY = 0.8  # Increased delay for scoring
MAX_RETRIES = 5
BATCH_SIZE = 30  # Save progress every N papers
MAX_TOKENS = 150

# Rate limiting and tracking
rate_limit_lock = Lock()
last_request_time = 0
request_counter = 0
error_counter = 0

# Score validation ranges
SCORE_RANGES = {
    "novelty": (0, 10),
    "technical_depth": (0, 10),
    "clarity": (0, 10),
    "impact_potential": (0, 10),
    "overall_score": (0.0, 10.0),
    "confidence": (0.0, 1.0)
}

def rate_limited_request():
    """Ensure minimum delay between API requests with jitter."""
    global last_request_time
    
    with rate_limit_lock:
        current_time = time.time()
        elapsed = current_time - last_request_time
        if elapsed < BASE_DELAY:
            sleep_time = BASE_DELAY - elapsed + random.uniform(0, 0.2)
            time.sleep(sleep_time)
        last_request_time = time.time()

def validate_scores(scores: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize score values."""
    if not isinstance(scores, dict):
        return {}
    
    validated = {}
    
    for key, (min_val, max_val) in SCORE_RANGES.items():
        if key in scores:
            try:
                value = scores[key]
                # Convert to appropriate type
                if key in ["overall_score", "confidence"]:
                    value = float(value)
                else:
                    value = int(round(float(value)))
                
                # Clamp to valid range
                if key in ["overall_score", "confidence"]:
                    value = max(min_val, min(max_val, value))
                else:
                    value = max(min_val, min(max_val, value))
                
                validated[key] = value
            except (ValueError, TypeError):
                logger.debug(f"Invalid value for {key}: {scores[key]}")
                # Set default values
                if key in ["overall_score", "confidence"]:
                    validated[key] = 0.0
                else:
                    validated[key] = 0
    
    # Calculate overall score if not provided but other scores exist
    if "overall_score" not in validated and all(k in validated for k in ["novelty", "technical_depth", "clarity", "impact_potential"]):
        validated["overall_score"] = round(
            (validated["novelty"] + validated["technical_depth"] + 
             validated["clarity"] + validated["impact_potential"]) / 4.0, 1
        )
    
    # Add confidence if missing
    if "confidence" not in validated:
        validated["confidence"] = 0.5
    
    return validated

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, KeyError, json.JSONDecodeError),
    max_tries=MAX_RETRIES,
    max_time=300,
    jitter=backoff.full_jitter
)
def call_deepseek_with_backoff(prompt: str, paper_id: str = "unknown") -> Optional[str]:
    """
    Call DeepSeek API with exponential backoff for scoring task.
    
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
                "temperature": 0.1,
                "max_tokens": MAX_TOKENS
            },
            timeout=REQUEST_TIMEOUT
        )
        
        # Handle rate limiting
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 90))
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
        if request_counter % 50 == 0:
            logger.info(f"Completed {request_counter} scoring requests")
        
        return content
        
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout for paper {paper_id}")
        raise
    except requests.exceptions.HTTPError as e:
        error_counter += 1
        status_code = e.response.status_code if e.response else "unknown"
        logger.error(f"HTTP error {status_code} for paper {paper_id}")
        if status_code >= 500:
            time.sleep(15)  # Longer sleep for server errors
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

def extract_paper_scores(paper: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract quality scores from paper metadata using DeepSeek API.
    
    Args:
        paper: Dictionary containing paper metadata
    
    Returns:
        Dictionary with original paper data plus quality scores
    """
    paper_id = paper.get("paper_id", paper.get("id", "unknown"))
    
    # Prepare content
    title = paper.get("title", "").strip()
    abstract = paper.get("abstract", "").strip()
    
    # Truncate if too long
    if len(abstract) > 2500:
        abstract = abstract[:2500] + "..."
    
    prompt = f"""
You are a senior conference reviewer with expertise in academic paper evaluation.

Evaluate the following paper strictly and objectively.

Score each dimension from 0 to 10 (integer only for novelty, technical_depth, clarity, impact_potential).
Overall score should be a float between 0.0 and 10.0 with one decimal place.
Confidence should be a float between 0.0 and 1.0 with two decimal places.

IMPORTANT: Return ONLY valid JSON in the following format, no additional text:
{{
  "novelty": int,
  "technical_depth": int,
  "clarity": int,
  "impact_potential": int,
  "overall_score": float,
  "confidence": float
}}

Evaluation Guidelines:
1. Novelty: Originality and new contributions (0=no novelty, 10=highly novel)
2. Technical Depth: Sophistication of methods and analysis (0=superficial, 10=very deep)
3. Clarity: Presentation quality and writing (0=unclear, 10=very clear)
4. Impact Potential: Potential influence on field (0=no impact, 10=high impact)
5. Overall Score: Weighted average considering all factors
6. Confidence: Your confidence in the evaluation (0.0=low, 1.0=high)

Title:
{title}

Abstract:
{abstract}
"""
    scores = {}
    
    try:
        api_response = call_deepseek(prompt, paper_id)
        
        if api_response is None:
            logger.debug(f"No API response for paper {paper_id}")
        else:
            # Clean response - remove markdown code blocks if present
            clean_response = api_response.strip()
            clean_response = re.sub(r'^```json\s*', '', clean_response)
            clean_response = re.sub(r'\s*```$', '', clean_response)
            
            # Try to parse as JSON
            try:
                parsed = json.loads(clean_response)
                
                if isinstance(parsed, dict):
                    scores = validate_scores(parsed)
                else:
                    logger.warning(f"Response is not a dictionary for paper {paper_id}")
                    
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON for paper {paper_id}: {str(e)[:100]}")
                
                # Advanced JSON extraction
                json_patterns = [
                    r'\{[^{}]*\}',  # Simple object
                    r'\{.*\}',      # Nested object (greedy)
                ]
                
                for pattern in json_patterns:
                    matches = re.findall(pattern, clean_response, re.DOTALL)
                    for match in matches:
                        try:
                            parsed = json.loads(match)
                            if isinstance(parsed, dict) and any(k in parsed for k in SCORE_RANGES.keys()):
                                scores = validate_scores(parsed)
                                if scores:
                                    break
                        except:
                            continue
                    if scores:
                        break
                
                # Fallback: extract scores from text
                if not scores:
                    logger.debug(f"Using text extraction fallback for paper {paper_id}")
                    
                    # Look for key-value patterns
                    for key in SCORE_RANGES.keys():
                        pattern = rf'"{key}"\s*:\s*([0-9]+(?:\.[0-9]+)?)'
                        match = re.search(pattern, clean_response, re.IGNORECASE)
                        if not match:
                            # Try without quotes
                            pattern = rf'{key}\s*:\s*([0-9]+(?:\.[0-9]+)?)'
                            match = re.search(pattern, clean_response, re.IGNORECASE)
                        
                        if match:
                            try:
                                value = match.group(1)
                                if key in ["overall_score", "confidence"]:
                                    scores[key] = float(value)
                                else:
                                    scores[key] = int(round(float(value)))
                            except:
                                pass
    
    except Exception as e:
        logger.error(f"Unexpected error scoring paper {paper_id}: {str(e)[:100]}")
    
    # Validate and add default values for missing scores
    scores = validate_scores(scores)
    
    # Create enhanced paper
    enhanced_paper = paper.copy()
    enhanced_paper["quality_scores"] = scores
    enhanced_paper["scoring_time"] = datetime.now().isoformat()
    
    return enhanced_paper

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
        "complete_scores": 0,
        "partial_scores": 0,
        "no_scores": 0
    }
    
    # Adjust workers for scoring task
    effective_workers = min(max_workers, 8)
    logger.info(f"Scoring {len(papers)} papers with {effective_workers} workers")
    
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        # Submit tasks
        future_to_index = {
            executor.submit(extract_paper_scores, paper): i 
            for i, paper in enumerate(papers)
        }
        
        # Track completion order for proper ordering
        results = [None] * len(papers)
        
        # Process with progress bar
        with tqdm(total=len(papers), desc="Evaluating papers", unit="paper") as pbar:
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                
                try:
                    enhanced_paper = future.result(timeout=REQUEST_TIMEOUT + 45)
                    results[idx] = enhanced_paper
                    
                    scores = enhanced_paper.get("quality_scores", {})
                    if scores:
                        required_keys = ["novelty", "technical_depth", "clarity", 
                                        "impact_potential", "overall_score", "confidence"]
                        present_keys = [k for k in required_keys if k in scores]
                        
                        if len(present_keys) >= 4:  # At least 4 of 6 scores
                            stats["complete_scores"] += 1
                            stats["successful"] += 1
                        elif len(present_keys) >= 1:
                            stats["partial_scores"] += 1
                            stats["successful"] += 1
                        else:
                            stats["no_scores"] += 1
                            stats["failed"] += 1
                    else:
                        stats["no_scores"] += 1
                        stats["failed"] += 1
                        
                except Exception as e:
                    logger.error(f"Processing failed for paper at index {idx}: {str(e)[:100]}")
                    stats["failed"] += 1
                    # Add fallback
                    fallback_paper = papers[idx].copy()
                    fallback_paper["quality_scores"] = {}
                    results[idx] = fallback_paper
                
                pbar.update(1)
                
                # Periodic checkpoint
                if checkpoint_path and pbar.n % BATCH_SIZE == 0:
                    # Collect completed results
                    completed = [r for r in results if r is not None]
                    save_checkpoint(completed, checkpoint_path)
                    
                    # Update progress in log
                    logger.info(f"Progress: {pbar.n}/{len(papers)} papers scored "
                               f"({pbar.n/len(papers)*100:.1f}%)")
    
    # Filter out None results
    enhanced_papers = [r for r in results if r is not None]
    
    return enhanced_papers, stats

def calculate_statistics(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics on the collected scores."""
    scores_data = []
    for paper in papers:
        scores = paper.get("quality_scores", {})
        if scores:
            scores_data.append(scores)
    
    if not scores_data:
        return {"total_papers_with_scores": 0}
    
    stats = {
        "total_papers_with_scores": len(scores_data),
        "average_scores": {},
        "score_distributions": {}
    }
    
    # Calculate averages
    for key in SCORE_RANGES.keys():
        if any(key in s for s in scores_data):
            values = [s[key] for s in scores_data if key in s]
            if values:
                stats["average_scores"][key] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
    
    # Calculate distributions for integer scores
    for key in ["novelty", "technical_depth", "clarity", "impact_potential"]:
        if key in [k for s in scores_data for k in s]:
            values = [s[key] for s in scores_data if key in s]
            distribution = {i: values.count(i) for i in range(11)}
            stats["score_distributions"][key] = distribution
    
    return stats

def main():
    """Main execution function."""
    
    # File paths
    input_file = Path("papers_final_aligned.jsonl")
    output_file = Path("papers_enhanced_scores.jsonl")
    checkpoint_file = Path("paper_scoring_checkpoint.jsonl")
    stats_file = Path("scoring_statistics.json")
    
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
        scoring_stats = calculate_statistics(enhanced_papers)
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(scoring_stats, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved scoring statistics to {stats_file}")
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")
        
        # Print summary
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info("=" * 70)
        logger.info("PAPER SCORING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total papers: {stats['total']}")
        logger.info(f"Successfully processed: {stats['successful']}")
        logger.info(f"Complete scores (4+ dimensions): {stats['complete_scores']}")
        logger.info(f"Partial scores (1-3 dimensions): {stats['partial_scores']}")
        logger.info(f"No scores: {stats['no_scores']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Total API requests: {request_counter}")
        logger.info(f"API errors: {error_counter}")
        if request_counter > 0:
            success_rate = (request_counter - error_counter) / request_counter * 100
            logger.info(f"API success rate: {success_rate:.1f}%")
        logger.info(f"Total time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        logger.info(f"Processing rate: {stats['total'] / elapsed_time:.2f} papers/sec")
        
        # Show score statistics if available
        if scoring_stats["total_papers_with_scores"] > 0:
            logger.info("-" * 70)
            logger.info("SCORING STATISTICS:")
            for key, val in scoring_stats["average_scores"].items():
                logger.info(f"  {key:20s}: {val['mean']:.2f} (min: {val['min']}, max: {val['max']})")
        
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
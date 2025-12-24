# enhance_keywords.py
"""
Enhanced academic keywords extraction using DeepSeek API with concurrent processing.

This script processes academic papers from a JSONL file, calls the DeepSeek API
to extract relevant academic keywords for each paper, and saves the enhanced data
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
        logging.FileHandler('keywords_extraction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API Configuration
API_KEY = "sk-79990d599cd74bc0a56f6ca2f200a621"
API_URL = "https://api.deepseek.com/v1/chat/completions"
REQUEST_TIMEOUT = 30
MAX_WORKERS = 8  # Adjust based on API rate limits
BASE_DELAY = 0.5  # Base delay between requests
MAX_RETRIES = 5
BATCH_SIZE = 50  # Save progress every N papers

# Rate limiting
rate_limit_lock = Lock()
last_request_time = 0
request_counter = 0
error_counter = 0

def rate_limited_request():
    """Ensure minimum delay between API requests with jitter."""
    global last_request_time
    
    with rate_limit_lock:
        current_time = time.time()
        elapsed = current_time - last_request_time
        if elapsed < BASE_DELAY:
            sleep_time = BASE_DELAY - elapsed + random.uniform(0, 0.1)
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
    Call DeepSeek API with exponential backoff and comprehensive error handling.
    
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
                "temperature": 0.3,
                "max_tokens": 300
            },
            timeout=REQUEST_TIMEOUT
        )
        
        # Handle rate limiting
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
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
        
        # Log success periodically
        if request_counter % 100 == 0:
            logger.info(f"Completed {request_counter} API requests")
        
        return content
        
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout for paper {paper_id}")
        raise
    except requests.exceptions.HTTPError as e:
        error_counter += 1
        status_code = e.response.status_code if e.response else "unknown"
        logger.error(f"HTTP error {status_code} for paper {paper_id}")
        if status_code >= 500:
            # Server error, use longer backoff
            time.sleep(10)
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

def extract_keywords(paper: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract academic keywords from paper metadata using DeepSeek API.
    
    Args:
        paper: Dictionary containing paper metadata
    
    Returns:
        Dictionary with original paper data plus extracted keywords
    """
    paper_id = paper.get("paper_id", paper.get("id", "unknown"))
    
    # Prepare title and abstract
    title = paper.get("title", "").strip()
    abstract = paper.get("abstract", "").strip()
    
    # Truncate if too long (API has token limits)
    if len(abstract) > 3000:
        abstract = abstract[:3000] + "..."
    
    prompt = f"""
Extract high-quality academic keywords from the following paper.

Rules:
- Output ONLY a JSON array of strings.
- 5 to 8 keywords.
- Keywords should describe tasks, methods, or research problems.
- Avoid generic words like "model", "method", "framework".
- Each keyword should be specific and meaningful.

Title:
{title}

Abstract:
{abstract}
"""
    keywords = []
    
    try:
        api_response = call_deepseek(prompt, paper_id)
        
        if api_response is None:
            logger.debug(f"No API response for paper {paper_id}")
        else:
            # Try to parse as JSON
            try:
                parsed = json.loads(api_response)
                
                if isinstance(parsed, list):
                    # Clean and validate keywords
                    for keyword in parsed:
                        if isinstance(keyword, str):
                            cleaned = keyword.strip()
                            # Filter out generic keywords and validate length
                            if (cleaned and 
                                len(cleaned) >= 3 and 
                                len(cleaned) <= 50 and
                                cleaned.lower() not in ['model', 'method', 'framework', 
                                                       'approach', 'system', 'algorithm']):
                                keywords.append(cleaned)
                    
                    # Deduplicate and limit
                    seen = set()
                    unique_keywords = []
                    for k in keywords:
                        if k not in seen:
                            seen.add(k)
                            unique_keywords.append(k)
                    keywords = unique_keywords[:8]  # Max 8 as per requirements
                    
                    # Ensure minimum of 5 if we have some
                    if 0 < len(keywords) < 5:
                        logger.debug(f"Paper {paper_id} has only {len(keywords)} keywords")
                
            except json.JSONDecodeError:
                # Fallback: extract keywords from text response
                logger.debug(f"Failed to parse JSON for paper {paper_id}, extracting from text")
                
                # Try to find JSON array pattern
                json_pattern = r'\[.*?\]'
                match = re.search(json_pattern, api_response, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                        if isinstance(parsed, list):
                            keywords = [str(k).strip() for k in parsed if str(k).strip()]
                            keywords = list(set(keywords))[:8]
                    except:
                        pass
                
                # If still no keywords, use text-based extraction
                if not keywords:
                    # Split by common delimiters and filter
                    lines = api_response.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        # Remove quotes, brackets, etc.
                        line = re.sub(r'^[\s\[\]\'"]+|[\s\[\]\'\"]+$', '', line)
                        if (line and 
                            len(line) >= 3 and 
                            len(line) <= 50 and
                            line.lower() not in ['model', 'method', 'framework'] and
                            ',' not in line and ';' not in line):
                            keywords.append(line)
                    
                    keywords = list(set(keywords))[:8]
    
    except Exception as e:
        logger.error(f"Unexpected error extracting keywords for paper {paper_id}: {str(e)[:100]}")
    
    # Create enhanced paper
    enhanced_paper = paper.copy()
    enhanced_paper["keywords"] = keywords
    enhanced_paper["keywords_extraction_time"] = datetime.now().isoformat()
    
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
                        checkpoint_path: Optional[Path] = None) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
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
        "with_keywords": 0,
        "empty_keywords": 0
    }
    
    # Adjust workers based on rate limits
    effective_workers = min(max_workers, 10)
    logger.info(f"Processing {len(papers)} papers with {effective_workers} workers")
    
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        # Submit tasks
        future_to_index = {
            executor.submit(extract_keywords, paper): i 
            for i, paper in enumerate(papers)
        }
        
        # Track completion order for proper ordering
        results = [None] * len(papers)
        
        # Process with progress bar
        with tqdm(total=len(papers), desc="Extracting keywords", unit="paper") as pbar:
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                
                try:
                    enhanced_paper = future.result(timeout=REQUEST_TIMEOUT + 30)
                    results[idx] = enhanced_paper
                    
                    keywords = enhanced_paper.get("keywords", [])
                    if keywords:
                        stats["with_keywords"] += 1
                        stats["successful"] += 1
                    else:
                        stats["empty_keywords"] += 1
                        stats["failed"] += 1
                        
                except Exception as e:
                    logger.error(f"Processing failed for paper at index {idx}: {str(e)[:100]}")
                    stats["failed"] += 1
                    # Add fallback
                    fallback_paper = papers[idx].copy()
                    fallback_paper["keywords"] = []
                    results[idx] = fallback_paper
                
                pbar.update(1)
                
                # Periodic checkpoint
                if checkpoint_path and pbar.n % BATCH_SIZE == 0:
                    # Collect completed results
                    completed = [r for r in results if r is not None]
                    save_checkpoint(completed, checkpoint_path)
    
    # Filter out None results (shouldn't happen with proper error handling)
    enhanced_papers = [r for r in results if r is not None]
    
    return enhanced_papers, stats

def main():
    """Main execution function."""
    
    # File paths
    input_file = Path("papers_final_aligned.jsonl")
    output_file = Path("papers_enhanced_keywords.jsonl")
    checkpoint_file = Path("keywords_extraction_checkpoint.jsonl")
    
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
        
        # Print summary
        elapsed_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("KEYWORDS EXTRACTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total papers: {stats['total']}")
        logger.info(f"Successfully processed: {stats['successful']}")
        logger.info(f"Papers with keywords: {stats['with_keywords']}")
        logger.info(f"Papers with empty keywords: {stats['empty_keywords']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Total API requests: {request_counter}")
        logger.info(f"API errors: {error_counter}")
        if request_counter > 0:
            logger.info(f"API success rate: {(request_counter - error_counter) / request_counter * 100:.1f}%")
        logger.info(f"Total time: {elapsed_time:.1f} seconds")
        logger.info(f"Processing rate: {stats['total'] / elapsed_time:.2f} papers/sec")
        logger.info("=" * 60)
        
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
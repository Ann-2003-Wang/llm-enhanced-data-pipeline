# enhance_summary.py

"""
Enhanced paper contribution summarization with optimized concurrent processing.

This script processes academic papers from a JSONL file, calls the DeepSeek API
to extract structured research contributions using concurrent processing with
adaptive rate limiting and request batching for maximum throughput.
"""

import json
import requests
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from tqdm import tqdm
import random
from datetime import datetime, timedelta
from threading import Lock, Semaphore
import re
import queue
from dataclasses import dataclass, field
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('contribution_extraction_optimized.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API Configuration
API_KEY = "sk-79990d599cd74bc0a56f6ca2f200a621"
API_URL = "https://api.deepseek.com/v1/chat/completions"
REQUEST_TIMEOUT = 35
MAX_WORKERS = 12  # Increased for higher concurrency
MIN_DELAY = 0.15  # Minimum delay between requests (optimized)
MAX_DELAY = 2.0   # Maximum delay for backoff
MAX_RETRIES = 4
BATCH_SIZE = 50
MAX_TOKENS = 350  # Optimized for faster responses

# Adaptive rate limiting
@dataclass
class RateLimiter:
    """Adaptive rate limiter that adjusts based on success/failure rates."""
    min_delay: float = MIN_DELAY
    max_delay: float = MAX_DELAY
    current_delay: float = MIN_DELAY
    success_window: deque = field(default_factory=lambda: deque(maxlen=50))
    lock: Lock = field(default_factory=Lock)
    
    def success(self):
        """Record a successful request."""
        with self.lock:
            self.success_window.append(True)
            # If last 10 requests had 90% success, decrease delay slightly
            if len(self.success_window) >= 10:
                success_rate = sum(1 for s in list(self.success_window)[-10:] if s) / 10
                if success_rate > 0.9 and self.current_delay > self.min_delay:
                    self.current_delay = max(self.min_delay, self.current_delay * 0.9)
    
    def failure(self):
        """Record a failed request."""
        with self.lock:
            self.success_window.append(False)
            # Increase delay on failure
            self.current_delay = min(self.max_delay, self.current_delay * 1.5)
    
    def get_delay(self) -> float:
        """Get current delay with jitter."""
        with self.lock:
            jitter = random.uniform(-0.05, 0.05) * self.current_delay
            return max(self.min_delay, self.current_delay + jitter)

# Global rate limiter and statistics
rate_limiter = RateLimiter()
request_counter = 0
error_counter = 0
last_request_time = 0
rate_lock = Lock()

# Expected contribution summary structure
CONTRIBUTION_STRUCTURE = {
    "problem": str,
    "method": str,
    "key_contributions": list,
    "application_scenarios": list
}

def adaptive_rate_limit():
    """Adaptive rate limiting with dynamic adjustment."""
    global last_request_time
    
    with rate_lock:
        current_time = time.time()
        elapsed = current_time - last_request_time
        delay_needed = rate_limiter.get_delay()
        
        if elapsed < delay_needed:
            sleep_time = delay_needed - elapsed
            time.sleep(sleep_time)
        
        last_request_time = time.time()
        return delay_needed

def call_deepseek_fast(prompt: str, paper_id: str = "unknown", timeout: int = REQUEST_TIMEOUT) -> Optional[str]:
    """
    Optimized API call with adaptive rate limiting and minimal overhead.
    
    Args:
        prompt: The prompt to send to the API
        paper_id: Paper identifier for logging
        timeout: Request timeout in seconds
    
    Returns:
        API response content as string, or None if failed
    """
    global request_counter, error_counter
    
    # Apply adaptive rate limiting
    adaptive_rate_limit()
    request_counter += 1
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.25,
        "max_tokens": MAX_TOKENS,
        "stream": False
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 429:
                # Rate limited - use exponential backoff
                wait_time = min(30, 2 ** attempt + random.uniform(0, 1))
                logger.warning(f"Rate limited for paper {paper_id}, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                rate_limiter.failure()
                continue
            
            response.raise_for_status()
            
            # Fast JSON parsing
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            if not content:
                raise ValueError("Empty response content")
            
            # Record success
            rate_limiter.success()
            
            # Return cleaned response
            return content.strip()
            
        except requests.exceptions.Timeout:
            logger.debug(f"Timeout for paper {paper_id}, attempt {attempt + 1}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(1 << attempt)  # Exponential backoff
            else:
                error_counter += 1
                rate_limiter.failure()
                return None
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code >= 500:
                logger.debug(f"Server error for paper {paper_id}: {e.response.status_code}")
                time.sleep(3 * (attempt + 1))
            else:
                logger.debug(f"HTTP error for paper {paper_id}: {e.response.status_code}")
                break  # Client error, don't retry immediately
            if attempt == MAX_RETRIES - 1:
                error_counter += 1
                rate_limiter.failure()
                return None
                
        except Exception as e:
            logger.debug(f"Request error for paper {paper_id}: {str(e)[:100]}")
            if attempt == MAX_RETRIES - 1:
                error_counter += 1
                rate_limiter.failure()
                return None
            time.sleep(0.5 * (attempt + 1))
    
    error_counter += 1
    rate_limiter.failure()
    return None

def validate_contribution_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Fast validation and cleaning of contribution summary."""
    if not isinstance(summary, dict):
        return {}
    
    validated = {}
    
    for field, expected_type in CONTRIBUTION_STRUCTURE.items():
        value = summary.get(field)
        
        if expected_type == str:
            if isinstance(value, str):
                validated[field] = value.strip()[:300]  # Limit length
            elif value is not None:
                validated[field] = str(value).strip()[:300]
            else:
                validated[field] = ""
        
        elif expected_type == list:
            if isinstance(value, list):
                # Fast cleaning of list items
                cleaned = []
                for item in value:
                    if isinstance(item, str) and item.strip():
                        cleaned.append(item.strip()[:200])
                    elif item is not None:
                        cleaned.append(str(item).strip()[:200])
                    if len(cleaned) >= 8:  # Limit to 8 items
                        break
                validated[field] = cleaned
            else:
                validated[field] = []
    
    return validated

def extract_contribution_summary_fast(paper: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fast extraction of contribution summary with optimized prompt.
    
    Args:
        paper: Dictionary containing paper metadata
    
    Returns:
        Dictionary with original paper data plus contribution summary
    """
    paper_id = paper.get("paper_id", paper.get("id", "unknown"))
    
    # Prepare content with length limits
    title = paper.get("title", "").strip()[:200]
    abstract = paper.get("abstract", "").strip()[:1500]
    
    # Optimized prompt for speed
    prompt = f"""Summarize this paper into JSON:

Title: {title}
Abstract: {abstract}

Return JSON with:
- "problem": research problem (1 sentence)
- "method": approach used (1 sentence)
- "key_contributions": list of 2-4 contributions
- "application_scenarios": list of 1-3 applications

JSON ONLY:"""
    
    summary = {}
    
    try:
        api_response = call_deepseek_fast(prompt, paper_id, timeout=25)
        
        if api_response:
            # Fast JSON extraction with fallback
            try:
                parsed = json.loads(api_response)
                summary = validate_contribution_summary(parsed)
            except json.JSONDecodeError:
                # Quick fallback: extract JSON-like structure
                json_match = re.search(r'\{[^{}]*\}', api_response, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                        summary = validate_contribution_summary(parsed)
                    except:
                        # Minimal text extraction
                        lines = [l.strip() for l in api_response.split('\n') if l.strip()]
                        if lines:
                            summary = {
                                "problem": lines[0][:200] if len(lines) > 0 else "",
                                "method": lines[1][:200] if len(lines) > 1 else "",
                                "key_contributions": lines[2:4] if len(lines) > 2 else [],
                                "application_scenarios": lines[4:6] if len(lines) > 4 else []
                            }
    except Exception as e:
        logger.debug(f"Error for paper {paper_id}: {str(e)[:80]}")
    
    # Create enhanced paper
    enhanced_paper = paper.copy()
    enhanced_paper["contribution_summary"] = summary
    
    return enhanced_paper

class BatchProcessor:
    """Batch processor for high-throughput concurrent processing."""
    
    def __init__(self, max_workers: int = MAX_WORKERS, batch_size: int = BATCH_SIZE):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.semaphore = Semaphore(max_workers * 2)  # Limit concurrent futures
        self.results_lock = Lock()
        
    def process_batch(self, papers: List[Dict[str, Any]], 
                     desc: str = "Processing") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process a batch of papers with optimized concurrency.
        
        Args:
            papers: List of papers to process
            desc: Description for progress bar
        
        Returns:
            Tuple of (processed papers, statistics)
        """
        stats = {
            "total": len(papers),
            "successful": 0,
            "failed": 0,
            "with_summary": 0
        }
        
        # Prepare results array
        results = [None] * len(papers)
        
        # Use ThreadPoolExecutor with optimized settings
        with ThreadPoolExecutor(max_workers=self.max_workers, 
                              thread_name_prefix="api_worker") as executor:
            
            # Submit all tasks
            futures = {}
            for idx, paper in enumerate(papers):
                with self.semaphore:
                    future = executor.submit(extract_contribution_summary_fast, paper)
                    futures[future] = idx
            
            # Process completed futures with progress bar
            completed = 0
            with tqdm(total=len(papers), desc=desc, unit="paper", 
                     mininterval=0.5, maxinterval=1.0) as pbar:
                
                for future in as_completed(futures):
                    idx = futures[future]
                    
                    try:
                        result = future.result(timeout=30)
                        results[idx] = result
                        
                        # Update statistics
                        summary = result.get("contribution_summary", {})
                        if summary:
                            stats["with_summary"] += 1
                            stats["successful"] += 1
                        else:
                            stats["failed"] += 1
                            
                    except Exception as e:
                        logger.debug(f"Batch processing error at index {idx}: {str(e)[:80]}")
                        stats["failed"] += 1
                        # Add fallback
                        fallback = papers[idx].copy()
                        fallback["contribution_summary"] = {}
                        results[idx] = fallback
                    
                    completed += 1
                    pbar.update(1)
                    
                    # Update progress periodically
                    if completed % 20 == 0:
                        pbar.set_postfix({
                            "rate": f"{completed/(time.time()-pbar.start_t):.1f}/s",
                            "delay": f"{rate_limiter.current_delay:.2f}s"
                        })
        
        # Filter out None results
        processed_papers = [r for r in results if r is not None]
        
        return processed_papers, stats

def load_papers_fast(input_path: Path) -> List[Dict[str, Any]]:
    """Fast loading of papers from JSONL file."""
    papers = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    papers.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
                    
        logger.info(f"Loaded {len(papers)} papers from {input_path}")
        return papers
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        raise

def save_checkpoint_fast(papers: List[Dict[str, Any]], checkpoint_path: Path):
    """Fast checkpoint saving."""
    try:
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            for paper in papers:
                f.write(json.dumps(paper, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.error(f"Checkpoint save error: {e}")

def load_checkpoint_fast(checkpoint_path: Path) -> Tuple[List[Dict[str, Any]], set]:
    """Fast checkpoint loading."""
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
        logger.error(f"Checkpoint load error: {e}")
    
    return papers, processed_ids

def main():
    """Main execution function with optimized processing."""
    
    # File paths
    input_file = Path("papers_final_aligned.jsonl")
    output_file = Path("papers_enhanced_contributions_fast.jsonl")
    checkpoint_file = Path("fast_contribution_checkpoint.jsonl")
    
    start_time = time.time()
    processor = BatchProcessor(max_workers=MAX_WORKERS, batch_size=BATCH_SIZE)
    
    try:
        # Load papers
        logger.info("Loading papers...")
        papers = load_papers_fast(input_file)
        
        if not papers:
            logger.error("No papers loaded. Exiting.")
            return
        
        # Load checkpoint
        checkpoint_papers, processed_ids = load_checkpoint_fast(checkpoint_file)
        
        # Determine remaining papers
        if processed_ids:
            papers_to_process = []
            for paper in papers:
                paper_id = paper.get("paper_id", paper.get("id"))
                if paper_id not in processed_ids:
                    papers_to_process.append(paper)
            
            logger.info(f"Resuming: {len(checkpoint_papers)} processed, {len(papers_to_process)} remaining")
            remaining_papers = papers_to_process
        else:
            logger.info(f"Starting fresh: {len(papers)} papers to process")
            remaining_papers = papers
        
        # Process in chunks for better memory management
        chunk_size = 500
        all_processed = []
        total_stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "with_summary": 0
        }
        
        for chunk_start in range(0, len(remaining_papers), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(remaining_papers))
            chunk = remaining_papers[chunk_start:chunk_end]
            
            logger.info(f"Processing chunk {chunk_start//chunk_size + 1}/{(len(remaining_papers)-1)//chunk_size + 1}")
            
            # Process chunk
            processed_chunk, stats = processor.process_batch(
                chunk, 
                desc=f"Chunk {chunk_start//chunk_size + 1}"
            )
            
            # Update statistics
            total_stats["total"] += stats["total"]
            total_stats["successful"] += stats["successful"]
            total_stats["failed"] += stats["failed"]
            total_stats["with_summary"] += stats["with_summary"]
            
            # Save checkpoint
            current_results = checkpoint_papers + all_processed + processed_chunk
            save_checkpoint_fast(current_results, checkpoint_file)
            
            # Add to results
            all_processed.extend(processed_chunk)
            
            # Show intermediate stats
            logger.info(f"Chunk complete: {stats['successful']}/{stats['total']} successful "
                       f"({stats['successful']/stats['total']*100:.1f}%)")
        
        # Combine all results
        enhanced_papers = checkpoint_papers + all_processed
        
        # Save final results
        logger.info("Saving final results...")
        save_checkpoint_fast(enhanced_papers, output_file)
        
        # Print comprehensive summary
        elapsed_time = time.time() - start_time
        papers_per_second = total_stats["total"] / elapsed_time
        
        logger.info("=" * 70)
        logger.info("OPTIMIZED PROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total papers processed: {total_stats['total']}")
        logger.info(f"Papers with summaries: {total_stats['with_summary']} "
                   f"({total_stats['with_summary']/total_stats['total']*100:.1f}%)")
        logger.info(f"Successful: {total_stats['successful']}")
        logger.info(f"Failed: {total_stats['failed']}")
        logger.info(f"Total API requests: {request_counter}")
        logger.info(f"API errors: {error_counter}")
        if request_counter > 0:
            success_rate = (request_counter - error_counter) / request_counter * 100
            logger.info(f"API success rate: {success_rate:.1f}%")
        logger.info(f"Final rate limit delay: {rate_limiter.current_delay:.2f}s")
        logger.info(f"Total time: {elapsed_time:.0f}s ({elapsed_time/60:.1f} min)")
        logger.info(f"Processing rate: {papers_per_second:.2f} papers/sec")
        logger.info(f"Estimated completion time for 10k papers: {10000/papers_per_second/60:.1f} min")
        logger.info(f"Output saved to: {output_file}")
        logger.info("=" * 70)
        
        # Remove checkpoint file
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                logger.info(f"Checkpoint removed: {checkpoint_file}")
            except Exception as e:
                logger.warning(f"Could not remove checkpoint: {e}")
        
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted. Checkpoint saved for resume.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)

if __name__ == "__main__":
    # Performance optimization: disable debug logging for production
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    main()
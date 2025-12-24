# enhance_fields_of_study.py
"""
Enhanced paper fields extraction using DeepSeek API with concurrent processing.

This script processes academic papers from a JSONL file, calls the DeepSeek API
to extract relevant academic fields of study for each paper, and saves the 
enhanced data to a new JSONL file with progress tracking and error handling.
"""

import json
import requests
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random
from datetime import datetime
import backoff

# Configure logging for better monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API Configuration
API_KEY = "sk-79990d599cd74bc0a56f6ca2f200a621"
API_URL = "https://api.deepseek.com/v1/chat/completions"
REQUEST_TIMEOUT = 30
MAX_WORKERS = 10  # Increased but with better rate limiting
BASE_DELAY = 1.0  # Base delay between requests
MAX_RETRIES = 5
BATCH_SIZE = 100  # Save progress every N papers

# Rate limiting and thread safety
rate_limit_lock = Lock()
last_request_time = 0
total_requests = 0
failed_requests = 0

def rate_limited_request():
    """Ensure minimum delay between API requests across all threads with jitter."""
    global last_request_time
    
    with rate_limit_lock:
        current_time = time.time()
        elapsed = current_time - last_request_time
        if elapsed < BASE_DELAY:
            sleep_time = BASE_DELAY - elapsed + random.uniform(0, 0.1)  # Add jitter
            time.sleep(sleep_time)
        last_request_time = time.time()

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, KeyError, json.JSONDecodeError),
    max_tries=MAX_RETRIES,
    max_time=300
)
def call_deepseek_api_with_backoff(prompt: str) -> Optional[str]:
    global total_requests, failed_requests
    
    # Apply rate limiting
    rate_limited_request()
    
    total_requests += 1
    
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
                "temperature": 0.2,
                "max_tokens": 200
            },
            timeout=REQUEST_TIMEOUT
        )
        
        # Check for rate limiting
        if response.status_code == 429:
            retry_after = response.headers.get('Retry-After', 60)
            logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(float(retry_after))
            response.raise_for_status()
        
        response.raise_for_status()
        
        data = response.json()
        
        # Validate response structure
        if not isinstance(data, dict):
            raise ValueError(f"Response is not a dictionary: {type(data)}")
        
        if "choices" not in data or not data["choices"]:
            raise ValueError(f"No choices in response: {data}")
        
        message = data["choices"][0].get("message", {})
        if not message or "content" not in message:
            raise ValueError(f"No content in message: {message}")
        
        content = message["content"].strip()
        
        # Validate that content is not empty
        if not content:
            raise ValueError("Empty response content")
        
        # Basic validation - should start with [ for JSON array
        if not content.startswith('[') or not content.endswith(']'):
            logger.warning(f"Response doesn't look like JSON array: {content[:100]}...")
            # Try to extract JSON array if it's wrapped in other text
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            else:
                raise ValueError(f"Response not a valid JSON array: {content[:100]}...")
        
        return content
        
    except requests.exceptions.Timeout:
        logger.warning("API request timeout")
        raise
    except requests.exceptions.HTTPError as e:
        failed_requests += 1
        if e.response.status_code >= 500:
            logger.error(f"Server error {e.response.status_code}: {e.response.text[:200]}")
        else:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text[:200]}")
        raise
    except Exception as e:
        failed_requests += 1
        logger.error(f"Unexpected API error: {str(e)[:200]}")
        raise

def call_deepseek_api(prompt: str, max_retries: int = MAX_RETRIES) -> Optional[str]:
    try:
        return call_deepseek_api_with_backoff(prompt)
    except Exception as e:
        logger.error(f"All retries failed for API call: {str(e)[:200]}")
        return None

def extract_academic_fields(paper: Dict[str, Any]) -> Dict[str, Any]:
    paper_id = paper.get("paper_id", paper.get("id", "unknown"))
    title = paper.get("title", "")[:200]  # Truncate for logging
    
    # Clean and prepare abstract
    abstract = paper.get("abstract", "")
    if abstract and len(abstract) > 1000:
        abstract = abstract[:1000] + "..."
    
    prompt = f"""
You are an academic classification expert.

Given the following paper information, infer the most appropriate academic fields of study.

Rules:
- Output ONLY a JSON array of strings.
- Each item must be a broad academic field (e.g., "Computer Vision", "Machine Learning").
- Include 3 to 6 fields only.
- Do NOT include model names or datasets.

Title:
{title}

Abstract:
{abstract}
"""
    fields = []
    
    try:
        api_response = call_deepseek_api(prompt)
        
        if api_response is None:
            logger.debug(f"No API response for paper {paper_id}")
        else:
            try:
                # Try to parse as JSON
                parsed = json.loads(api_response)
                
                if isinstance(parsed, list):
                    # Validate and clean each field
                    for field in parsed:
                        if isinstance(field, str):
                            cleaned = field.strip()
                            if cleaned and len(cleaned) < 100:  # Reasonable length check
                                fields.append(cleaned)
                
                # Limit to 6 fields
                fields = fields[:6]
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON for paper {paper_id}: {str(e)[:100]}")
                logger.debug(f"Raw response: {api_response[:500]}")
                
                # Fallback: try to extract fields from text response
                if api_response:
                    import re
                    # Look for field-like patterns
                    potential_fields = re.findall(r'"([^"]+)"', api_response)
                    if not potential_fields:
                        potential_fields = re.findall(r'[\w\s]+(?=,|\.|$)', api_response)
                    
                    fields = [f.strip() for f in potential_fields if len(f.strip()) > 3 and len(f.strip()) < 50]
                    fields = list(set(fields))[:6]  # Deduplicate and limit
    
    except Exception as e:
        logger.error(f"Unexpected error extracting fields for paper {paper_id}: {str(e)[:200]}")
    
    # Create a copy of the paper with extracted fields
    enhanced_paper = paper.copy()
    enhanced_paper["fields_of_study"] = fields
    enhanced_paper["fields_extraction_time"] = datetime.now().isoformat()
    
    return enhanced_paper

def load_papers(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    papers = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                papers.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON on line {line_num}: {e}")
    
    logger.info(f"Loaded {len(papers)} papers from {input_path}")
    return papers

def save_checkpoint(papers: List[Dict[str, Any]], checkpoint_path: Path):
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        for paper in papers:
            f.write(json.dumps(paper, ensure_ascii=False) + '\n')
    
    logger.info(f"Checkpoint saved: {len(papers)} papers to {checkpoint_path}")

def load_checkpoint(checkpoint_path: Path) -> Tuple[List[Dict[str, Any]], set]:
    if not checkpoint_path.exists():
        return [], set()
    
    papers = []
    processed_ids = set()
    
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
    return papers, processed_ids

def process_paper_batch(papers: List[Dict[str, Any]], 
                       max_workers: int = MAX_WORKERS,
                       checkpoint_path: Optional[Path] = None) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    enhanced_papers = []
    stats = {
        "successful": 0,
        "failed": 0,
        "empty_fields": 0,
        "total": len(papers)
    }
    
    # Adjust workers based on rate limits
    effective_workers = min(max_workers, 5)  # Conservative start
    
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        # Submit all tasks
        future_to_paper = {
            executor.submit(extract_academic_fields, paper): paper 
            for paper in papers
        }
        
        # Process results as they complete
        logger.info(f"Processing {len(papers)} papers with {effective_workers} concurrent workers...")
        
        completed = 0
        for future in tqdm(as_completed(future_to_paper), total=len(papers), 
                          desc="Extracting academic fields", unit="paper"):
            try:
                enhanced_paper = future.result(timeout=REQUEST_TIMEOUT + 10)
                enhanced_papers.append(enhanced_paper)
                
                fields = enhanced_paper.get("fields_of_study", [])
                if fields:
                    stats["successful"] += 1
                else:
                    stats["empty_fields"] += 1
                    stats["failed"] += 1
                    
            except Exception as e:
                logger.error(f"Error processing paper: {str(e)[:200]}")
                stats["failed"] += 1
                # Add original paper with empty fields as fallback
                original_paper = future_to_paper[future].copy()
                original_paper["fields_of_study"] = []
                enhanced_papers.append(original_paper)
            
            completed += 1
            
            # Save checkpoint periodically
            if checkpoint_path and completed % BATCH_SIZE == 0:
                save_checkpoint(enhanced_papers, checkpoint_path)
    
    return enhanced_papers, stats

def main():
    """Main execution function with concurrent processing and checkpointing."""
    
    # Define file paths
    input_file = Path("../Data_Cleaning/papers_final_aligned.jsonl")
    output_file = Path("papers_enhanced_fields.jsonl")
    checkpoint_file = Path("papers_enhanced_checkpoint.jsonl")
    
    start_time = time.time()
    
    try:
        # Load papers
        papers = load_papers(input_file)
        
        if not papers:
            logger.error("No papers loaded from input file")
            return
        
        # Load checkpoint if exists
        checkpoint_papers, processed_ids = load_checkpoint(checkpoint_file)
        
        # Filter out already processed papers
        if processed_ids:
            papers_to_process = []
            for paper in papers:
                paper_id = paper.get("paper_id", paper.get("id"))
                if paper_id not in processed_ids:
                    papers_to_process.append(paper)
            
            logger.info(f"Already processed {len(checkpoint_papers)} papers. "
                       f"Processing {len(papers_to_process)} remaining papers.")
            
            # Process remaining papers
            enhanced_new_papers, stats = process_paper_batch(
                papers_to_process, 
                checkpoint_path=checkpoint_file
            )
            
            # Combine checkpoint and new results
            enhanced_papers = checkpoint_papers + enhanced_new_papers
        else:
            # Process all papers
            enhanced_papers, stats = process_paper_batch(
                papers, 
                checkpoint_path=checkpoint_file
            )
        
        # Final save
        with open(output_file, 'w', encoding='utf-8') as f:
            for paper in enhanced_papers:
                f.write(json.dumps(paper, ensure_ascii=False) + '\n')
        
        # Print summary
        elapsed_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"PROCESSING COMPLETE!")
        logger.info(f"Total papers processed: {stats['total']}")
        logger.info(f"Successfully extracted fields: {stats['successful']}")
        logger.info(f"Papers with empty fields: {stats['empty_fields']}")
        logger.info(f"Failed extractions: {stats['failed']}")
        logger.info(f"Total API requests: {total_requests}")
        logger.info(f"Failed API requests: {failed_requests}")
        logger.info(f"Success rate: {(total_requests - failed_requests) / total_requests * 100:.1f}%")
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Processing rate: {stats['total'] / elapsed_time:.2f} papers/second")
        logger.info(f"Results saved to: {output_file}")
        logger.info("=" * 60)
        
        # Clean up checkpoint file
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info(f"Checkpoint file removed: {checkpoint_file}")
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user.")
        logger.info(f"Checkpoint saved to {checkpoint_file}. Resume by running again.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
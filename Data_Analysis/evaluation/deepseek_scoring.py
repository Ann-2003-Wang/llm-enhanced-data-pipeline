# deepseek_scoring.py
import json
import asyncio
import aiohttp
import logging
from typing import Dict, Any, List
import time
from tqdm.asyncio import tqdm
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================
INPUT_FILE = "merged_papers.jsonl"
OUTPUT_FILE = "papers_scored_raw.jsonl"

# Use environment variable or hardcoded key
DEEPSEEK_API_KEY = "sk-79990d599cd74bc0a56f6ca2f200a621"  # Replace with your API key
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

MODEL_NAME = "deepseek-chat"
MAX_CONCURRENCY = 5  # Reduced for API stability
REQUEST_DELAY = 0.5  # Delay between requests in seconds
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

# =========================
# Prompt templates
# =========================
SYSTEM_PROMPT = """You are an expert academic data curator and NLP researcher.
Your task is to evaluate the quality of an academic paper record
based ONLY on the provided title, abstract, and metadata.

You must NOT assume any missing information.
If information is missing or unclear, assign a LOWER score."""

USER_PROMPT_TEMPLATE = """
Evaluate the following academic paper record.

Title:
{title}

Abstract:
{abstract}

Metadata:
- Authors: {authors}
- Publish Year: {publish_year}
- Venue: {venue}
- Fields of Study: {fields_of_study}
- URL: {url}

Scoring dimensions (each 0–5):

1. Metadata Completeness
2. Text Cleanliness
3. Technical Specificity
4. Domain Relevance (AI / ML / CV / NLP / Robotics)
5. Semantic Clarity
6. Downstream Usability for retrieval / RAG systems

Rules:
- Be strict and conservative.
- Missing or empty fields MUST reduce scores.
- Do NOT infer citation impact or popularity.
- Base judgment ONLY on the given text.

Return ONLY valid JSON in the following format:

{{
  "metadata_completeness": int,
  "text_cleanliness": int,
  "technical_specificity": int,
  "domain_relevance": int,
  "semantic_clarity": int,
  "downstream_usability": int,
  "overall_score": float
}}
"""

# =========================
# Utils
# =========================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dictionaries."""
    papers = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    papers.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: JSON decode error - {e}")
    except FileNotFoundError:
        logger.error(f"Input file not found: {path}")
        raise
    
    logger.info(f"Loaded {len(papers)} papers from {path}")
    return papers

def clean_text(text: Any, max_length: int = 500) -> str:
    """Clean text field."""
    if text is None:
        return ""
    if isinstance(text, str):
        return text.strip()[:max_length]
    return str(text).strip()[:max_length]

def build_user_prompt(paper: Dict[str, Any]) -> str:
    """Build user prompt from paper data."""
    # Clean and truncate fields to avoid excessive token usage
    title = clean_text(paper.get("title"), 300)
    abstract = clean_text(paper.get("abstract"), 1500)
    
    # Handle authors
    authors = paper.get("authors", [])
    if isinstance(authors, list):
        authors_str = ", ".join([clean_text(a) for a in authors[:3]])
    else:
        authors_str = clean_text(authors)
    
    # Other fields
    publish_year = clean_text(paper.get("publish_year"), 10)
    venue = clean_text(paper.get("venue"), 100)
    
    # Fields of study
    fields = paper.get("fields_of_study", [])
    if isinstance(fields, list):
        fields_str = ", ".join([clean_text(f) for f in fields[:3]])
    else:
        fields_str = clean_text(fields)
    
    url = clean_text(paper.get("url"), 200)
    
    return USER_PROMPT_TEMPLATE.format(
        title=title,
        abstract=abstract,
        authors=authors_str,
        publish_year=publish_year,
        venue=venue,
        fields_of_study=fields_str,
        url=url,
    )

# =========================
# API Call with retry
# =========================
async def call_deepseek_with_retry(
    session: aiohttp.ClientSession,
    prompt: str,
    paper_id: str = "unknown",
    max_retries: int = MAX_RETRIES
) -> Dict[str, Any]:
    """Call DeepSeek API with retry logic."""
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 300,
    }
    
    for attempt in range(max_retries):
        try:
            # Small delay between retries
            if attempt > 0:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            async with session.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout
            ) as resp:
                # Get response text
                response_text = await resp.text()
                
                # Check for rate limiting
                if resp.status == 429:
                    logger.warning(f"Paper {paper_id}: Rate limited (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(10)  # Wait longer for rate limits
                        continue
                    else:
                        raise RuntimeError("Rate limited after all retries")
                
                # Check for other HTTP errors
                if resp.status != 200:
                    logger.error(f"Paper {paper_id}: HTTP error {resp.status}: {response_text[:200]}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        raise RuntimeError(f"HTTP error {resp.status}")
                
                # Parse JSON response
                try:
                    data = json.loads(response_text)
                except json.JSONDecodeError:
                    logger.error(f"Paper {paper_id}: Invalid JSON response: {response_text[:200]}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        raise RuntimeError("Invalid JSON response")
                
                # Extract content
                try:
                    content = data["choices"][0]["message"]["content"].strip()
                    
                    # Try to parse scores JSON
                    try:
                        scores = json.loads(content)
                    except json.JSONDecodeError:
                        # Try to find JSON in the response
                        import re
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            scores = json.loads(json_match.group(0))
                        else:
                            raise ValueError(f"No JSON found in response: {content[:100]}")
                    
                    # Validate required keys
                    required_keys = [
                        "metadata_completeness", 
                        "text_cleanliness", 
                        "technical_specificity",
                        "domain_relevance", 
                        "semantic_clarity", 
                        "downstream_usability",
                        "overall_score"
                    ]
                    
                    for key in required_keys:
                        if key not in scores:
                            scores[key] = 0
                    
                    # Calculate normalized score (0-10)
                    total_possible = 30  # 6 dimensions * 5 max each
                    scores["overall_score_normalized"] = round(
                        scores["overall_score"] / total_possible * 10, 2
                    )
                    
                    return scores
                    
                except (KeyError, ValueError) as e:
                    logger.error(f"Paper {paper_id}: Response parsing error: {e}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        raise RuntimeError(f"Response parsing error: {e}")
        
        except asyncio.TimeoutError:
            logger.warning(f"Paper {paper_id}: Timeout (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                continue
            else:
                raise RuntimeError("Request timeout after all retries")
        
        except aiohttp.ClientError as e:
            logger.error(f"Paper {paper_id}: Network error: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                raise RuntimeError(f"Network error: {e}")
    
    # If all retries failed
    raise RuntimeError("All retries failed")

# =========================
# Paper scoring function
# =========================
async def score_paper(
    session: aiohttp.ClientSession,
    paper: Dict[str, Any],
    idx: int
) -> Dict[str, Any]:
    """Score a single paper."""
    
    paper_id = paper.get("paper_id", paper.get("id", f"paper_{idx}"))
    
    try:
        # Build prompt
        prompt = build_user_prompt(paper)
        
        # Call API
        scores = await call_deepseek_with_retry(session, prompt, paper_id)
        
        # Add scores to paper copy
        paper_copy = paper.copy()
        paper_copy["quality_scores"] = scores
        paper_copy["scored"] = True
        
        return paper_copy
        
    except Exception as e:
        logger.error(f"Paper {paper_id}: Failed to score - {str(e)[:100]}")
        
        # Return paper with error scores
        paper_copy = paper.copy()
        paper_copy["quality_scores"] = {
            "metadata_completeness": 0,
            "text_cleanliness": 0,
            "technical_specificity": 0,
            "domain_relevance": 0,
            "semantic_clarity": 0,
            "downstream_usability": 0,
            "overall_score": 0.0,
            "overall_score_normalized": 0.0,
            "error": str(e)
        }
        paper_copy["scored"] = False
        
        return paper_copy

# =========================
# Main processing
# =========================
async def process_papers_concurrently(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process all papers concurrently with rate limiting."""
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    
    # Create aiohttp session with connection pooling
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENCY * 2)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT * 3)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        
        # Create tasks
        tasks = []
        for idx, paper in enumerate(papers):
            # Create async task with semaphore control
            async def process_with_semaphore(paper, idx):
                async with semaphore:
                    # Add small delay between requests to avoid rate limiting
                    await asyncio.sleep(REQUEST_DELAY)
                    return await score_paper(session, paper, idx)
            
            task = asyncio.create_task(process_with_semaphore(paper, idx))
            tasks.append(task)
        
        # Process results with progress bar
        results = []
        task_iterator = asyncio.as_completed(tasks)
        
        # Use tqdm for progress tracking
        pbar = tqdm(total=len(tasks), desc="Scoring papers", unit="paper")
        
        for task in task_iterator:
            try:
                result = await task
                results.append(result)
                pbar.update(1)
                
                # Log progress every 50 papers
                if len(results) % 50 == 0:
                    logger.info(f"Progress: {len(results)}/{len(papers)} papers scored")
                    
            except Exception as e:
                logger.error(f"Task processing error: {e}")
                # Add placeholder for failed task
                paper_idx = len(results)
                if paper_idx < len(papers):
                    paper_copy = papers[paper_idx].copy()
                    paper_copy["quality_scores"] = {"error": str(e)}
                    paper_copy["scored"] = False
                    results.append(paper_copy)
                    pbar.update(1)
        
        pbar.close()
    
    return results

# =========================
# Save results
# =========================
def save_results(papers: List[Dict[str, Any]], output_path: str):
    """Save scored papers to JSONL file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for paper in papers:
                f.write(json.dumps(paper, ensure_ascii=False) + "\n")
        
        # Calculate statistics
        successful = sum(1 for p in papers if p.get("scored", False))
        failed = len(papers) - successful
        
        logger.info(f"Saved {len(papers)} papers to {output_path}")
        logger.info(f"Successfully scored: {successful}")
        logger.info(f"Failed: {failed}")
        
        if successful > 0:
            # Calculate average scores
            valid_scores = [p["quality_scores"] for p in papers if p.get("scored", False)]
            avg_scores = {}
            
            for key in ["metadata_completeness", "text_cleanliness", "technical_specificity",
                       "domain_relevance", "semantic_clarity", "downstream_usability"]:
                values = [s.get(key, 0) for s in valid_scores if key in s]
                if values:
                    avg_scores[key] = sum(values) / len(values)
            
            overall_values = [s.get("overall_score_normalized", 0) for s in valid_scores]
            if overall_values:
                avg_scores["overall_score_normalized"] = sum(overall_values) / len(overall_values)
            
            logger.info("Average scores (successful papers only):")
            for key, value in avg_scores.items():
                logger.info(f"  {key}: {value:.2f}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

# =========================
# Jupyter compatibility
# =========================
def run_async_in_jupyter():
    """Run async function in Jupyter environment."""
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # In Jupyter, we need to use nest_asyncio
            try:
                import nest_asyncio
                nest_asyncio.apply()
                logger.info("Applied nest_asyncio for Jupyter compatibility")
            except ImportError:
                logger.warning("nest_asyncio not installed. Install with: pip install nest_asyncio")
                # Create new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
    except RuntimeError:
        # No event loop exists, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return asyncio.get_event_loop()

# =========================
# Main function
# =========================
async def main_async():
    """Main async function."""
    logger.info("Starting paper scoring...")
    logger.info(f"Concurrency: {MAX_CONCURRENCY}, Delay: {REQUEST_DELAY}s")
    
    start_time = time.time()
    
    # Load papers
    papers = load_jsonl(INPUT_FILE)
    if not papers:
        logger.error("No papers to process. Exiting.")
        return []
    
    logger.info(f"Processing {len(papers)} papers")
    
    # Process papers concurrently
    results = await process_papers_concurrently(papers)
    
    # Save results
    save_results(results, OUTPUT_FILE)
    
    # Final statistics
    elapsed_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total papers: {len(papers)}")
    logger.info(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}min)")
    logger.info(f"Processing rate: {len(papers)/elapsed_time:.2f} papers/sec")
    logger.info("=" * 60)
    
    return results

def main():
    """Main entry point compatible with both script and Jupyter."""
    try:
        # Check if we're in Jupyter
        in_jupyter = 'IPython' in sys.modules
        
        if in_jupyter:
            logger.info("Running in Jupyter environment")
            loop = run_async_in_jupyter()
            results = loop.run_until_complete(main_async())
        else:
            # Regular script execution
            results = asyncio.run(main_async())
        
        return results
        
    except KeyboardInterrupt:
        logger.info("\nScoring interrupted by user")
        return []
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return []

if __name__ == "__main__":
    main()
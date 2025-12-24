"""
================================================================================
Enhanced Quality Scoring Module for Academic Papers
================================================================================

Purpose:
This module provides advanced quality assessment for academic papers by implementing
a multi-dimensional scoring system that evaluates papers based on completeness,
recency, academic substance, methodological rigor, impact potential, and domain
relevance. It is designed to process cleaned papers and generate enhanced quality
scores with confidence metrics and detailed statistical analysis.

Key Features:
1. MULTI-DIMENSIONAL QUALITY ASSESSMENT:
   - Completeness Score (15%): Evaluates presence and quality of key metadata fields
   - Recency Score (20%): Assesses publication timeliness using exponential decay
   - Academic Substance Score (30%): Measures technical depth and innovation
   - Methodological Rigor Score (25%): Evaluates research methodology and reproducibility
   - Impact Potential Score (10%): Assesses practical applications and influence potential
   - Domain Relevance Score (15%): Measures alignment with target research topics

2. ADVANCED SCORING ALGORITHMS:
   - Exponential decay for recency scoring to prioritize recent publications
   - Technical term extraction for academic substance evaluation
   - Heuristic pattern matching for methodological rigor assessment
   - Penalty system for low-quality or suspicious content
   - Confidence scoring based on data completeness and reliability

3. COMPREHENSIVE STATISTICAL ANALYSIS:
   - Detailed quality score distribution across six tiers (Excellent to Low)
   - Correlation analysis between quality scores and text length/recency
   - Mean, median, standard deviation, min/max score calculations
   - Preservation of original quality scores for comparison

4. INTELLIGENT DOMAIN ADAPTATION:
   - Dynamic domain term matching for relevance scoring
   - Common AI/ML/CV/Robo/NLP terminology library for baseline comparison
   - Category-based relevance assessment for computer science papers

5. OUTPUT AND REPORTING:
   - Enhanced JSONL output with comprehensive quality metadata
   - Detailed quality statistics in JSON format
   - Console reporting of top-performing papers
   - Tier-based distribution analysis

Input Requirements:
- cleaned_papers.jsonl (output from original cleaning pipeline)
  This file should contain papers with cleaned text and basic metadata.

Output Files:
1. enhanced_scored_papers.jsonl - Papers with enhanced quality scores
2. quality_statistics.json - Comprehensive quality score statistics

Scoring Methodology:
Each paper receives a total score (0-1) based on weighted components:
- Total Score = Σ(Component_Score × Weight) - Penalty
- Quality Tiers: Excellent (≥0.8), High (0.7-0.8), Good (0.6-0.7), 
  Medium (0.5-0.6), Fair (0.4-0.5), Low (<0.4)

Penalty System:
Penalties are applied for:
- Short abstracts (<100 characters: 0.2, <50 characters: 0.4)
- Short titles (<10 characters: 0.1)
- Suspicious content patterns (e.g., draft indicators)
- Duplicate content (0.3 penalty)

Confidence Scoring:
Confidence (0-1) is calculated based on:
- Data completeness (normalized completeness score)
- Text length (longer texts = higher confidence)
- Technical term density (more terms = higher confidence)

Usage:
Run as standalone script: python quality_scoring.py
Or integrate into pipeline via main() function.

Note:
This module is designed to be run AFTER the cleaning pipeline and BEFORE
provenance_compliance.py in the data processing workflow.
================================================================================
"""

import json
import re
import numpy as np
from datetime import datetime
import statistics
from typing import Dict, List, Tuple

def load_jsonl_file(filepath: str) -> List[Dict]:
    """Load JSONL file"""
    papers = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            papers.append(json.loads(line.strip()))
    return papers

def enhanced_quality_scoring(paper: Dict, topic_terms: List[str] = None) -> Dict:
    """Enhanced quality scoring for academic papers"""
    if topic_terms is None:
        topic_terms = []
    
    scores = {}
    
    # 1. Basic quality scores
    scores['completeness'] = enhanced_score_completeness(paper) * 0.15
    scores['recency'] = enhanced_score_recency(paper) * 0.20
    
    # 2. Academic quality scores
    scores['academic_substance'] = score_academic_substance(paper) * 0.30
    scores['methodological_rigor'] = score_methodological_rigor(paper) * 0.25
    scores['impact_potential'] = score_impact_potential(paper) * 0.10
    
    # 3. Domain relevance score
    scores['domain_relevance'] = score_domain_relevance(paper, topic_terms) * 0.15
    
    # Calculate weighted total score
    total_score = sum(scores.values())
    
    # 4. Penalty calculation
    penalty = calculate_penalty(paper)
    total_score = max(0, total_score - penalty)
    
    # 5. Confidence calculation
    confidence = calculate_confidence(paper, scores)
    
    return {
        'scores': scores,
        'total_score': round(total_score, 3),
        'confidence': confidence,
        'quality_tier': calculate_quality_tier(total_score),
        'penalty_applied': penalty
    }

def enhanced_score_completeness(paper: Dict) -> float:
    """Enhanced completeness scoring"""
    score = 0.0
    mandatory_fields = ['title', 'abstract', 'authors']
    optional_fields = ['categories', 'comments', 'publish_date', 'url']
    
    # Check mandatory fields
    for field in mandatory_fields:
        if paper.get(field):
            if field == 'abstract' and len(paper['abstract']) > 100:
                score += 0.3
            elif field == 'title' and len(paper['title']) > 10:
                score += 0.25
            elif field == 'authors' and len(paper['authors']) > 0:
                score += 0.25
        else:
            # Penalty for missing mandatory fields
            return 0.0
    
    # Check optional fields
    for field in optional_fields:
        if paper.get(field):
            if field == 'categories' and len(paper['categories']) > 0:
                score += 0.05
            elif field == 'comments' and len(paper['comments']) > 0:
                score += 0.05
            elif field in ['publish_date', 'url']:
                score += 0.05
    
    return min(1.0, score)

def enhanced_score_recency(paper: Dict) -> float:
    """Enhanced recency scoring using exponential decay"""
    if 'publish_date' not in paper:
        return 0.3
    
    try:
        publish_date = datetime.strptime(paper['publish_date'], '%Y-%m-%d').date()
        current_date = datetime.now().date()
        days_ago = (current_date - publish_date).days
        
        # Exponential decay scoring
        if days_ago <= 7:
            return 1.0
        elif days_ago <= 30:
            return 0.9 - (days_ago - 7) * 0.01
        elif days_ago <= 90:
            return 0.7 - (days_ago - 30) * 0.005
        elif days_ago <= 365:
            return 0.4 - (days_ago - 90) * 0.001
        else:
            return max(0.1, 0.3 - (days_ago - 365) * 0.0001)
    except Exception:
        return 0.3

def score_academic_substance(paper: Dict) -> float:
    """Academic substance scoring"""
    abstract = paper.get('abstract', '').lower()
    title = paper.get('title', '').lower()
    
    score = 0.0
    
    # Technical term density
    tech_terms = paper.get('technical_terms', [])
    tech_count = len(tech_terms) if tech_terms else 0
    if tech_count >= 5:
        score += 0.4
    elif tech_count >= 3:
        score += 0.25
    elif tech_count >= 1:
        score += 0.1
    
    # Innovation indicators
    innovation_indicators = ['novel', 'new', 'propose', 'introduce', 'innovative', 
                           'original', 'state-of-the-art', 'sota', 'breakthrough']
    innovation_count = sum(1 for word in innovation_indicators if word in abstract)
    if innovation_count >= 3:
        score += 0.3
    elif innovation_count >= 2:
        score += 0.2
    elif innovation_count >= 1:
        score += 0.1
    
    # Experimental evaluation indicators
    eval_indicators = ['experiment', 'evaluation', 'benchmark', 'comparison', 
                      'result', 'performance', 'accuracy', 'precision', 'recall']
    eval_count = sum(1 for word in eval_indicators if word in abstract)
    if eval_count >= 3:
        score += 0.2
    elif eval_count >= 2:
        score += 0.15
    elif eval_count >= 1:
        score += 0.1
    
    # Theoretical/mathematical indicators
    theory_indicators = ['theorem', 'proof', 'lemma', 'corollary', 'equation',
                        'formula', 'mathematical', 'theoretical']
    if any(indicator in abstract for indicator in theory_indicators):
        score += 0.1
    
    return min(1.0, score)

def score_methodological_rigor(paper: Dict) -> float:
    """Methodological rigor scoring"""
    abstract = paper.get('abstract', '').lower()
    
    score = 0.0
    
    # Method description completeness
    method_descriptors = ['method', 'approach', 'framework', 'algorithm', 'model',
                         'architecture', 'technique', 'strategy']
    method_count = sum(1 for word in method_descriptors if word in abstract)
    if method_count >= 2:
        score += 0.3
    elif method_count >= 1:
        score += 0.15
    
    # Evaluation metrics mention
    metrics = ['metric', 'measure', 'score', 'f1', 'roc', 'auc', 'mse', 'mae']
    if any(metric in abstract for metric in metrics):
        score += 0.2
    
    # Dataset/benchmark mention
    dataset_indicators = ['dataset', 'benchmark', 'corpus', 'collection', 'db']
    if any(indicator in abstract for indicator in dataset_indicators):
        score += 0.2
    
    # Statistical significance indicators
    stat_indicators = ['significant', 'p-value', 'confidence', 'interval', 'variance']
    if any(indicator in abstract for indicator in stat_indicators):
        score += 0.15
    
    # Reproducibility indicators
    reproducibility = ['reproducible', 'replication', 'code available', 'github']
    if any(indicator in abstract for indicator in reproducibility):
        score += 0.15
    
    return min(1.0, score)

def score_impact_potential(paper: Dict) -> float:
    """Impact potential scoring"""
    abstract = paper.get('abstract', '').lower()
    title = paper.get('title', '').lower()
    
    score = 0.0
    
    # Application potential
    application_indicators = ['application', 'real-world', 'practical', 'deploy',
                            'industry', 'clinical', 'commercial', 'usable']
    if any(indicator in abstract for indicator in application_indicators):
        score += 0.3
    
    # General applicability
    general_indicators = ['general', 'universal', 'broad', 'wide', 'scalable']
    if any(indicator in abstract for indicator in general_indicators):
        score += 0.2
    
    # Citation potential (via terminology)
    impactful_terms = ['foundation', 'fundamental', 'paradigm', 'landmark', 
                      'seminal', 'pioneering', 'transformative']
    if any(term in abstract for term in impactful_terms):
        score += 0.25
    
    # Open source/resource contribution
    resource_indicators = ['open source', 'resource', 'toolkit', 'library',
                          'package', 'platform', 'system']
    if any(indicator in abstract for indicator in resource_indicators):
        score += 0.25
    
    return min(1.0, score)

def score_domain_relevance(paper: Dict, topic_terms: List[str]) -> float:
    """Domain relevance scoring"""
    if not topic_terms:
        return 0.5
    
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    
    # Calculate domain term match ratio
    matched_terms = [term.lower() for term in topic_terms if term.lower() in text]
    match_ratio = len(matched_terms) / len(topic_terms) if topic_terms else 0
    
    # Category relevance
    categories = paper.get('categories', [])
    cs_categories = [cat for cat in categories if 'cs.' in cat.lower()]
    category_score = 0.3 if len(cs_categories) > 0 else 0.1
    
    # Domain terms in title
    title = paper.get('title', '').lower()
    title_matches = sum(1 for term in topic_terms if term.lower() in title)
    title_score = min(0.2, title_matches * 0.1)
    
    return min(1.0, match_ratio * 0.5 + category_score + title_score)

def calculate_penalty(paper: Dict) -> float:
    """Calculate penalty score"""
    penalty = 0.0
    
    # Short abstract penalty
    abstract = paper.get('abstract', '')
    if len(abstract) < 100:
        penalty += 0.2
    elif len(abstract) < 50:
        penalty += 0.4
    
    # Short title penalty
    title = paper.get('title', '')
    if len(title) < 10:
        penalty += 0.1
    
    # Suspicious content penalty (based on heuristic rules)
    suspicious_patterns = [
        r'\b(arxiv|submit|preprint)\b.*?\b(version|draft)\b',
        r'\b(this paper|we study|we investigate)\b.*?\b(without)\b.*?\b(result|experiment)\b'
    ]
    
    text = f"{title} {abstract}".lower()
    for pattern in suspicious_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            penalty += 0.15
            break
    
    # Duplicate content penalty (simple check)
    if 'duplicate_of' in paper:
        penalty += 0.3
    
    return min(0.5, penalty)  # Maximum penalty 0.5

def calculate_confidence(paper: Dict, scores: Dict) -> float:
    """Calculate scoring confidence"""
    confidence_factors = []
    
    # Confidence based on data completeness
    completeness = scores.get('completeness', 0) / 0.15  # Normalized
    confidence_factors.append(completeness)
    
    # Confidence based on text length
    abstract_len = len(paper.get('abstract', ''))
    if abstract_len > 500:
        confidence_factors.append(1.0)
    elif abstract_len > 200:
        confidence_factors.append(0.8)
    elif abstract_len > 50:
        confidence_factors.append(0.5)
    else:
        confidence_factors.append(0.2)
    
    # Confidence based on term count
    tech_terms = paper.get('technical_terms', [])
    if len(tech_terms) >= 3:
        confidence_factors.append(1.0)
    elif len(tech_terms) >= 1:
        confidence_factors.append(0.7)
    else:
        confidence_factors.append(0.3)
    
    return round(statistics.mean(confidence_factors), 3)

def calculate_quality_tier(score: float) -> str:
    """Calculate quality tier based on score"""
    if score >= 0.8:
        return 'Excellent'
    elif score >= 0.7:
        return 'High'
    elif score >= 0.6:
        return 'Good'
    elif score >= 0.5:
        return 'Medium'
    elif score >= 0.4:
        return 'Fair'
    else:
        return 'Low'

def generate_quality_statistics(papers: List[Dict]) -> Dict:
    """Generate quality score statistics"""
    if not papers:
        return {}
    
    scores = [p.get('enhanced_quality', {}).get('total_score', 0) for p in papers]
    
    return {
        'summary': {
            'total_papers': len(papers),
            'mean_score': round(np.mean(scores), 3),
            'median_score': round(np.median(scores), 3),
            'std_deviation': round(np.std(scores), 3),
            'min_score': round(min(scores), 3) if scores else 0,
            'max_score': round(max(scores), 3) if scores else 0
        },
        'distribution': {
            'excellent': sum(1 for p in papers 
                           if p.get('enhanced_quality', {}).get('quality_tier') == 'Excellent'),
            'high': sum(1 for p in papers 
                       if p.get('enhanced_quality', {}).get('quality_tier') == 'High'),
            'good': sum(1 for p in papers 
                       if p.get('enhanced_quality', {}).get('quality_tier') == 'Good'),
            'medium': sum(1 for p in papers 
                         if p.get('enhanced_quality', {}).get('quality_tier') == 'Medium'),
            'fair': sum(1 for p in papers 
                       if p.get('enhanced_quality', {}).get('quality_tier') == 'Fair'),
            'low': sum(1 for p in papers 
                      if p.get('enhanced_quality', {}).get('quality_tier') == 'Low')
        },
        'correlations': {
            'score_vs_length': calculate_correlation_score_length(papers),
            'score_vs_recency': calculate_correlation_score_recency(papers)
        }
    }

def calculate_correlation_score_length(papers: List[Dict]) -> float:
    """Calculate correlation between score and text length"""
    scores = []
    lengths = []
    
    for paper in papers:
        score = paper.get('enhanced_quality', {}).get('total_score', 0)
        length = len(paper.get('abstract', '')) + len(paper.get('title', ''))
        scores.append(score)
        lengths.append(length)
    
    if len(scores) > 1:
        correlation = np.corrcoef(scores, lengths)[0, 1]
        return round(correlation, 3)
    return 0.0

def calculate_correlation_score_recency(papers: List[Dict]) -> float:
    """Calculate correlation between score and recency"""
    scores = []
    recency = []
    
    for paper in papers:
        score = paper.get('enhanced_quality', {}).get('total_score', 0)
        if 'publish_date' in paper:
            try:
                publish_date = datetime.strptime(paper['publish_date'], '%Y-%m-%d')
                days_ago = (datetime.now() - publish_date).days
                recency.append(1.0 / (days_ago + 1))  # Higher value = more recent
                scores.append(score)
            except:
                continue
    
    if len(scores) > 1:
        correlation = np.corrcoef(scores, recency)[0, 1]
        return round(correlation, 3)
    return 0.0

def save_scored_papers(papers: List[Dict], output_file: str = "scored_papers.jsonl"):
    """Save papers with enhanced quality scores"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for paper in papers:
            f.write(json.dumps(paper, ensure_ascii=False) + '\n')
    print(f"✓ Saved {len(papers)} papers with enhanced quality scores to {output_file}")

def main():
    """Main function"""
    print("=" * 60)
    print("ENHANCED QUALITY SCORING SYSTEM")
    print("=" * 60)
    
    # Load cleaned papers data
    input_file = "cleaned_papers.jsonl"
    print(f"Loading {input_file}...")
    
    try:
        papers = load_jsonl_file(input_file)
        print(f"Successfully loaded {len(papers)} papers")
    except FileNotFoundError:
        print(f"Error: File not found {input_file}")
        print("Please run original code first to generate cleaned_papers.jsonl")
        return
    
    # Perform enhanced quality scoring for each paper
    print("Performing enhanced quality scoring...")
    
    # Get domain terms (simplified approach, should be based on paper topics)
    # Using common AI/ML/CV terms as examples
    common_ai_terms = [
        'machine learning', 'deep learning', 'neural network', 
        'artificial intelligence', 'computer vision', 'natural language processing',
        'reinforcement learning', 'supervised learning', 'unsupervised learning',
        'convolutional neural network', 'transformer', 'attention mechanism',
        'generative adversarial network', 'autoencoder', 'decision tree',
        'support vector machine', 'random forest', 'gradient boosting'
    ]
    
    for i, paper in enumerate(papers, 1):
        # Preserve original quality scores (if exist)
        if 'quality_scores' in paper:
            paper['original_quality_scores'] = paper['quality_scores']
            paper['original_overall_score'] = paper.get('overall_quality_score', 0)
            paper['original_quality_tier'] = paper.get('quality_tier', 'unknown')
        
        # Perform enhanced quality scoring
        enhanced_scores = enhanced_quality_scoring(paper, common_ai_terms)
        paper['enhanced_quality'] = enhanced_scores
        
        if i % 10 == 0:
            print(f"  Processed {i}/{len(papers)} papers...")
    
    print("✓ Quality scoring completed")
    
    # Generate statistics
    print("\nGenerating quality statistics...")
    stats = generate_quality_statistics(papers)
    
    # Save statistics
    stats_file = "quality_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"✓ Quality statistics saved to {stats_file}")
    
    # Save scored papers
    output_file = "enhanced_scored_papers.jsonl"
    save_scored_papers(papers, output_file)
    
    # Display summary statistics
    print("\n" + "=" * 60)
    print("QUALITY SCORING SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total papers: {stats['summary']['total_papers']}")
    print(f"Mean score: {stats['summary']['mean_score']}")
    print(f"Highest score: {stats['summary']['max_score']}")
    print(f"Lowest score: {stats['summary']['min_score']}")
    print("\nQuality tier distribution:")
    for tier, count in stats['distribution'].items():
        percentage = (count / stats['summary']['total_papers']) * 100
        print(f"  {tier}: {count} papers ({percentage:.1f}%)")
    
    print(f"\nScore vs text length correlation: {stats['correlations']['score_vs_length']}")
    print(f"Score vs recency correlation: {stats['correlations']['score_vs_recency']}")
    
    # Display top 3 highest-scoring papers
    print("\n" + "=" * 60)
    print("TOP 3 HIGHEST-SCORING PAPERS")
    print("=" * 60)
    
    sorted_papers = sorted(papers, 
                         key=lambda x: x.get('enhanced_quality', {}).get('total_score', 0), 
                         reverse=True)
    
    for i, paper in enumerate(sorted_papers[:3], 1):
        score = paper.get('enhanced_quality', {}).get('total_score', 0)
        tier = paper.get('enhanced_quality', {}).get('quality_tier', 'Unknown')
        print(f"\n{i}. {paper.get('title', 'No Title')[:80]}...")
        print(f"   Score: {score:.3f} ({tier})")
        print(f"   Authors: {', '.join(paper.get('authors', ['Unknown']))[:60]}...")
        print(f"   Date: {paper.get('publish_date', 'Unknown')}")
    
    print("\n" + "=" * 60)
    print("✓ Enhanced quality scoring process completed")
    print(f"✓ Output file: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
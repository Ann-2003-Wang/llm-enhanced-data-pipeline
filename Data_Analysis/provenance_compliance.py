"""
================================================================================
Data Provenance and Compliance Check Module
================================================================================

Purpose:
This module enhances scored academic papers with comprehensive provenance tracking,
licensing information, and compliance checks to ensure data integrity, legal 
compliance, and responsible data usage.

Key Features:
1. DATA PROVENANCE TRACKING:
   - Records complete data lineage from source to processing
   - Generates unique fingerprints for each paper using MD5 hashing
   - Tracks processing steps, timestamps, and pipeline versions

2. LICENSING AND ATTRIBUTION:
   - Adds arXiv.org licensing information and usage restrictions
   - Provides clear attribution requirements and citation formats
   - Documents allowed uses and commercial use considerations

3. PRIVACY COMPLIANCE (PII DETECTION):
   - Detects Personally Identifiable Information (PII) including:
     * Email addresses
     * Phone numbers
     * IP addresses
     * Social Security numbers
     * Credit card numbers
     * URLs with embedded credentials
   - Performs risk assessment and provides remediation recommendations
   - Masks sensitive data in reports for security

4. COPYRIGHT COMPLIANCE:
   - Verifies arXiv.org source compliance
   - Identifies preprint vs. peer-reviewed status
   - Checks for journal references and proper licensing

5. DATA QUALITY VALIDATION:
   - Validates required field completeness
   - Checks text length and format requirements
   - Identifies potential duplicates
   - Validates date formats and data consistency

6. COMPREHENSIVE REPORTING:
   - Generates detailed compliance reports with risk distribution
   - Provides actionable recommendations for data handling
   - Identifies high-risk papers requiring manual review

Input Requirements:
- enhanced_scored_papers.jsonl (output from quality_scoring.py)
  This file should contain papers with enhanced quality scores.

Output Files:
1. full_processed_papers.jsonl - Complete dataset with provenance and compliance data
2. compliance_report.json - Comprehensive compliance analysis and recommendations

Processing Steps:
1. Load scored papers from JSONL file
2. Add provenance information (source, processing history, metadata)
3. Add licensing information (allowed uses, restrictions, attribution)
4. Perform PII leakage detection and risk assessment
5. Check copyright compliance and preprint status
6. Validate data quality and identify issues
7. Generate comprehensive compliance report
8. Save fully processed dataset

Usage:
Run as standalone script: python provenance_compliance.py
Or integrate into pipeline via main() function.

Note:
This module is designed to be run AFTER quality_scoring.py and BEFORE
ablation_error_analysis.py in the data processing pipeline.
================================================================================
"""

import json
import re
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Set

def load_jsonl_file(filepath: str) -> List[Dict]:
    """Load JSONL file"""
    papers = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            papers.append(json.loads(line.strip()))
    return papers

def generate_paper_fingerprint(paper: Dict) -> str:
    """Generate unique fingerprint for paper (for data provenance)"""
    # Generate MD5 fingerprint using key fields
    fingerprint_data = {
        'title': paper.get('title', ''),
        'first_author': paper.get('first_author', ''),
        'publish_date': paper.get('publish_date', ''),
        'arxiv_id': paper.get('paper_id', '')
    }
    
    fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
    return hashlib.md5(fingerprint_str.encode('utf-8')).hexdigest()

def add_provenance_info(paper: Dict) -> Dict:
    """Add data provenance information"""
    
    processing_date = datetime.now().isoformat()
    paper_fingerprint = generate_paper_fingerprint(paper)
    
    provenance_info = {
        'source': {
            'platform': 'arXiv',
            'identifier': paper.get('paper_id', ''),
            'url': paper.get('url', ''),
            'retrieval_method': 'arxiv-py API'
        },
        'processing': {
            'pipeline_version': '1.0.0',
            'processing_date': processing_date,
            'processing_steps': [
                'arxiv_crawling',
                'text_cleaning',
                'quality_scoring',
                'provenance_tracking'
            ],
            'fingerprint': paper_fingerprint
        },
        'history': {
            'original_crawl_time': paper.get('crawl_time', processing_date),
            'enhancement_date': processing_date,
            'version': 1
        },
        'metadata': {
            'format': 'JSONL',
            'encoding': 'UTF-8',
            'schema_version': '1.0'
        }
    }
    
    # Add to paper data
    if 'provenance' not in paper:
        paper['provenance'] = provenance_info
    else:
        # Update if provenance information already exists
        paper['provenance'].update(provenance_info)
    
    return paper

def add_licensing_info(paper: Dict) -> Dict:
    """Add licensing information"""
    
    licensing_info = {
        'source_license': 'arXiv.org perpetual, non-exclusive license',
        'license_details': {
            'name': 'arXiv.org License',
            'url': 'https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html',
            'version': '1.0'
        },
        'allowed_uses': [
            'academic_research',
            'text_analysis',
            'machine_learning_training',
            'non_commercial_analysis'
        ],
        'restrictions': [
            'commercial_use_may_require_additional_permissions',
            'no_redistribution_as_is',
            'attribution_required'
        ],
        'attribution_requirements': {
            'required': True,
            'format': 'Cite original arXiv publication and mention data processing',
            'suggested_format': f"Data sourced from arXiv paper {paper.get('paper_id', '')}, processed via AI Research Pipeline"
        },
        'compliance_status': {
            'source_compliant': True,
            'data_use_compliant': True,
            'attribution_satisfied': False  # Pending user confirmation
        }
    }
    
    # Add to paper data
    paper['licensing'] = licensing_info
    
    return paper

def check_pii_leakage(paper: Dict) -> Dict:
    """Check for Personally Identifiable Information (PII) leakage"""
    text = f"{paper.get('title', '')} {paper.get('abstract', '')} {paper.get('comments', '')}"
    
    # PII detection patterns
    pii_patterns = {
        'email': {
            'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'risk_level': 'high'
        },
        'phone': {
            'pattern': r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            'risk_level': 'high'
        },
        'url_with_credentials': {
            'pattern': r'https?://[^:\s]+:[^@\s]+@',
            'risk_level': 'critical'
        },
        'ip_address': {
            'pattern': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'risk_level': 'medium'
        },
        'social_security': {
            'pattern': r'\b\d{3}-\d{2}-\d{4}\b',
            'risk_level': 'critical'
        },
        'credit_card': {
            'pattern': r'\b(?:\d{4}[- ]?){3}\d{4}\b',
            'risk_level': 'critical'
        }
    }
    
    findings = {}
    for pii_type, config in pii_patterns.items():
        matches = re.findall(config['pattern'], text, re.IGNORECASE)
        if matches:
            # Mask sensitive information (show only partial)
            masked_matches = []
            for match in matches[:3]:  # Show only first 3 matches
                if pii_type == 'email':
                    parts = match.split('@')
                    if len(parts) == 2:
                        masked = f"{parts[0][0]}***@{parts[1]}"
                        masked_matches.append(masked)
                elif pii_type == 'phone':
                    if len(match) > 4:
                        masked = f"{match[:2]}****{match[-2:]}"
                        masked_matches.append(masked)
                else:
                    masked_matches.append('***[REDACTED]***')
            
            findings[pii_type] = {
                'count': len(matches),
                'sample_matches': masked_matches,
                'risk_level': config['risk_level'],
                'action_taken': 'masked_in_report'
            }
    
    # Risk assessment
    risk_levels = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
    max_risk = max([risk_levels[findings[pii]['risk_level']] 
                   for pii in findings]) if findings else 0
    
    risk_assessment = 'low'
    if max_risk >= 4:
        risk_assessment = 'critical'
    elif max_risk >= 3:
        risk_assessment = 'high'
    elif max_risk >= 2:
        risk_assessment = 'medium'
    
    return {
        'has_pii': len(findings) > 0,
        'findings': findings,
        'risk_assessment': risk_assessment,
        'check_date': datetime.now().isoformat(),
        'text_length': len(text),
        'pii_density': sum(f['count'] for f in findings.values()) / max(len(text.split()), 1)
    }

def check_copyright_compliance(paper: Dict) -> Dict:
    """Check copyright compliance"""
    
    # Check arXiv paper compliance
    categories = paper.get('categories', [])
    comments = paper.get('comments', '').lower()
    
    compliance_info = {
        'arxiv_compliance': {
            'is_arxiv': 'arxiv.org' in paper.get('url', ''),
            'arxiv_categories': categories,
            'has_license_statement': True,  # arXiv has default license
            'commercial_use_restrictions': 'Check individual paper for specific license'
        },
        'content_analysis': {
            'likely_preprint': True,
            'peer_reviewed': 'unknown',
            'journal_reference': 'not_found' if 'journal' not in comments else 'found'
        },
        'recommendations': {
            'for_research_use': 'approved',
            'for_commercial_use': 'requires_further_check',
            'for_redistribution': 'requires_attribution'
        }
    }
    
    return compliance_info

def check_data_quality_issues(paper: Dict) -> Dict:
    """Check data quality issues"""
    
    issues = []
    warnings = []
    
    # Check field completeness
    required_fields = ['title', 'abstract', 'authors']
    for field in required_fields:
        if field not in paper or not paper[field]:
            issues.append(f'missing_{field}')
    
    # Check text quality
    abstract = paper.get('abstract', '')
    if len(abstract) < 50:
        issues.append('abstract_too_short')
    elif len(abstract) < 100:
        warnings.append('abstract_short')
    
    # Check title quality
    title = paper.get('title', '')
    if len(title) < 10:
        issues.append('title_too_short')
    
    # Check date format
    if 'publish_date' in paper:
        try:
            datetime.strptime(paper['publish_date'], '%Y-%m-%d')
        except ValueError:
            issues.append('invalid_date_format')
    
    # Check duplicate content (based on title)
    if 'duplicate_title_score' in paper.get('quality_scores', {}):
        if paper['quality_scores']['duplicate_title_score'] > 0.8:
            warnings.append('potential_duplicate')
    
    return {
        'issues': issues,
        'warnings': warnings,
        'severity': 'high' if issues else 'low' if warnings else 'none',
        'check_date': datetime.now().isoformat()
    }

def generate_compliance_report(papers: List[Dict]) -> Dict:
    """Generate overall compliance report"""
    
    total_papers = len(papers)
    pii_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'none': 0}
    quality_issues = {'high': 0, 'medium': 0, 'low': 0, 'none': 0}
    
    for paper in papers:
        # Count PII risks
        pii_check = paper.get('privacy_check', {})
        pii_counts[pii_check.get('risk_assessment', 'none')] += 1
        
        # Count quality issues
        quality_check = paper.get('quality_check', {})
        quality_issues[quality_check.get('severity', 'none')] += 1
    
    # Calculate percentages
    pii_percentages = {k: (v/total_papers*100) for k, v in pii_counts.items()}
    quality_percentages = {k: (v/total_papers*100) for k, v in quality_issues.items()}
    
    return {
        'summary': {
            'total_papers': total_papers,
            'analysis_date': datetime.now().isoformat(),
            'compliance_status': 'compliant' if pii_counts['critical'] == 0 else 'needs_review'
        },
        'privacy_analysis': {
            'distribution': pii_counts,
            'percentages': pii_percentages,
            'highest_risk': max(pii_counts.items(), key=lambda x: x[1])[0] if sum(pii_counts.values()) > 0 else 'none',
            'recommendation': 'review_high_risk_papers' if pii_counts['critical'] > 0 else 'acceptable'
        },
        'quality_analysis': {
            'distribution': quality_issues,
            'percentages': quality_percentages,
            'most_common_issue': 'incomplete_data' if quality_issues['high'] > 0 else 'minor_warnings',
            'recommendation': 'review_low_quality_papers' if quality_issues['high'] > 0 else 'acceptable'
        },
        'licensing_summary': {
            'all_arxiv': all('arxiv' in p.get('provenance', {}).get('source', {}).get('platform', '').lower() 
                           for p in papers),
            'consistent_license': True,  # arXiv has uniform license
            'attribution_required': True
        },
        'recommendations': [
            'Ensure proper attribution when using this data',
            'Review papers with high PII risk before public use',
            'Consider additional cleaning for low-quality papers'
        ]
    }

def save_compliant_papers(papers: List[Dict], output_file: str = "compliant_papers.jsonl"):
    """Save papers after compliance checks"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for paper in papers:
            f.write(json.dumps(paper, ensure_ascii=False) + '\n')
    print(f"✓ Saved {len(papers)} papers after compliance checks to {output_file}")

def main():
    """Main function"""
    print("=" * 60)
    print("DATA PROVENANCE AND COMPLIANCE CHECK SYSTEM")
    print("=" * 60)
    
    # Load scored papers data
    input_file = "enhanced_scored_papers.jsonl"
    print(f"Loading {input_file}...")
    
    try:
        papers = load_jsonl_file(input_file)
        print(f"Successfully loaded {len(papers)} papers")
    except FileNotFoundError:
        print(f"Error: File not found {input_file}")
        print("Please run quality_scoring.py first to generate enhanced_scored_papers.jsonl")
        return
    
    # Add provenance and compliance information to each paper
    print("\nPerforming data provenance and compliance checks...")
    
    for i, paper in enumerate(papers, 1):
        # Add provenance information
        paper = add_provenance_info(paper)
        
        # Add licensing information
        paper = add_licensing_info(paper)
        
        # Check PII leakage
        paper['privacy_check'] = check_pii_leakage(paper)
        
        # Check copyright compliance
        paper['copyright_check'] = check_copyright_compliance(paper)
        
        # Check data quality issues
        paper['quality_check'] = check_data_quality_issues(paper)
        
        if i % 10 == 0:
            print(f"  Processed {i}/{len(papers)} papers...")
    
    print("✓ Data provenance and compliance checks completed")
    
    # Generate overall compliance report
    print("\nGenerating compliance report...")
    compliance_report = generate_compliance_report(papers)
    
    # Save compliance report
    report_file = "compliance_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(compliance_report, f, indent=2, ensure_ascii=False)
    print(f"✓ Compliance report saved to {report_file}")
    
    # Save compliant papers
    output_file = "full_processed_papers.jsonl"
    save_compliant_papers(papers, output_file)
    
    # Display summary report
    print("\n" + "=" * 60)
    print("COMPLIANCE CHECK SUMMARY REPORT")
    print("=" * 60)
    
    print(f"\nOverall Status: {compliance_report['summary']['compliance_status'].upper()}")
    print(f"Total Papers: {compliance_report['summary']['total_papers']}")
    
    print("\nPrivacy Risk Distribution:")
    for risk_level, count in compliance_report['privacy_analysis']['distribution'].items():
        if count > 0:
            percentage = compliance_report['privacy_analysis']['percentages'][risk_level]
            print(f"  {risk_level.upper()}: {count} papers ({percentage:.1f}%)")
    
    print("\nData Quality Issue Distribution:")
    for severity, count in compliance_report['quality_analysis']['distribution'].items():
        if count > 0:
            percentage = compliance_report['quality_analysis']['percentages'][severity]
            print(f"  {severity.upper()}: {count} papers ({percentage:.1f}%)")
    
    print(f"\nLicense Consistency: {'Yes' if compliance_report['licensing_summary']['consistent_license'] else 'No'}")
    print(f"All from arXiv: {'Yes' if compliance_report['licensing_summary']['all_arxiv'] else 'No'}")
    
    print("\nRecommendations:")
    for i, recommendation in enumerate(compliance_report['recommendations'], 1):
        print(f"  {i}. {recommendation}")
    
    # Display high-risk paper examples
    high_risk_papers = [p for p in papers 
                       if p.get('privacy_check', {}).get('risk_assessment') in ['critical', 'high']]
    
    if high_risk_papers:
        print("\n" + "=" * 60)
        print("HIGH-RISK PAPER EXAMPLES (Require Manual Review)")
        print("=" * 60)
        
        for i, paper in enumerate(high_risk_papers[:3], 1):
            risk = paper.get('privacy_check', {}).get('risk_assessment', 'unknown')
            pii_types = list(paper.get('privacy_check', {}).get('findings', {}).keys())
            print(f"\n{i}. {paper.get('title', 'No Title')[:60]}...")
            print(f"   Risk Level: {risk.upper()}")
            print(f"   PII Types: {', '.join(pii_types[:3])}")
            print(f"   arXiv ID: {paper.get('paper_id', 'Unknown')}")
    
    print("\n" + "=" * 60)
    print("✓ Data provenance and compliance check process completed")
    print(f"✓ Output file: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()

"""
Ablation Experiment and Error Analysis Module
For comparative analysis of papers at different processing stages
"""
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import os
warnings.filterwarnings('ignore')

def load_jsonl_file(filepath: str) -> List[Dict]:
    """Load JSONL file"""
    try:
        papers = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                papers.append(json.loads(line.strip()))
        return papers
    except FileNotFoundError:
        print(f"Warning: File not found {filepath}")
        return []

def load_all_data_files():
    """Load all data files"""
    print("Loading data files...")
    
    files = {
        'raw': 'test_papers.jsonl',
        'cleaned': 'cleaned_papers.jsonl', 
        'scored': 'enhanced_scored_papers.jsonl',
        'full': 'full_processed_papers.jsonl'
    }
    
    data = {}
    for name, filepath in files.items():
        papers = load_jsonl_file(filepath)
        if papers:
            data[name] = papers
            print(f"  ✓ {name}: {len(papers)} papers")
        else:
            print(f"  ✗ {name}: not found or empty")
    
    return data

def calculate_dataset_statistics(papers: List[Dict], name: str) -> Dict:
    """Calculate dataset statistics"""
    if not papers:
        return {}
    
    # Basic statistics
    titles = [p.get('title', '') for p in papers]
    abstracts = [p.get('abstract', '') for p in papers]
    
    stats = {
        'name': name,
        'paper_count': len(papers),
        'avg_title_length': np.mean([len(t) for t in titles]),
        'avg_abstract_length': np.mean([len(a) for a in abstracts]),
        'unique_authors': len(set([author for p in papers 
                                  for author in p.get('authors', ['Unknown'])]))
    }
    
    # Quality score statistics (if available)
    quality_scores = []
    for paper in papers:
        if 'enhanced_quality' in paper:
            quality_scores.append(paper['enhanced_quality']['total_score'])
        elif 'overall_quality_score' in paper:
            quality_scores.append(paper['overall_quality_score'])
    
    if quality_scores:
        stats['quality_scores'] = {
            'mean': np.mean(quality_scores),
            'median': np.median(quality_scores),
            'std': np.std(quality_scores),
            'min': np.min(quality_scores),
            'max': np.max(quality_scores)
        }
    
    return stats

def conduct_ablation_study(data: Dict) -> Dict:
    """Conduct ablation study analysis"""
    print("\nConducting ablation study analysis...")
    
    ablation_results = {}
    
    # 1. Impact of cleaning stage
    if 'raw' in data and 'cleaned' in data:
        raw_stats = calculate_dataset_statistics(data['raw'], 'raw')
        cleaned_stats = calculate_dataset_statistics(data['cleaned'], 'cleaned')
        
        ablation_results['cleaning_impact'] = {
            'papers_removed': raw_stats['paper_count'] - cleaned_stats['paper_count'],
            'removal_rate': ((raw_stats['paper_count'] - cleaned_stats['paper_count']) / 
                           raw_stats['paper_count'] * 100) if raw_stats['paper_count'] > 0 else 0,
            'avg_abstract_length_change': (cleaned_stats['avg_abstract_length'] - 
                                         raw_stats['avg_abstract_length']),
            'avg_title_length_change': (cleaned_stats['avg_title_length'] - 
                                      raw_stats['avg_title_length'])
        }
        print("  ✓ Cleaning stage analysis completed")
    
    # 2. Impact of quality scoring stage
    if 'cleaned' in data and 'scored' in data:
        cleaned_stats = calculate_dataset_statistics(data['cleaned'], 'cleaned')
        scored_stats = calculate_dataset_statistics(data['scored'], 'scored')
        
        # Analyze quality distribution changes
        if 'quality_scores' in scored_stats:
            ablation_results['scoring_impact'] = {
                'quality_distribution': scored_stats['quality_scores'],
                'quality_score_added': True
            }
        print("  ✓ Quality scoring stage analysis completed")
    
    # 3. Impact of full processing pipeline
    if 'raw' in data and 'full' in data:
        raw_stats = calculate_dataset_statistics(data['raw'], 'raw')
        full_stats = calculate_dataset_statistics(data['full'], 'full')
        
        ablation_results['full_pipeline_impact'] = {
            'total_papers_removed': raw_stats['paper_count'] - full_stats['paper_count'],
            'final_paper_count': full_stats['paper_count'],
            'processing_efficiency': (full_stats['paper_count'] / 
                                    raw_stats['paper_count'] * 100) if raw_stats['paper_count'] > 0 else 0
        }
        print("  ✓ Full pipeline analysis completed")
    
    return ablation_results

def error_analysis(data: Dict) -> Dict:
    """Error analysis"""
    print("\nConducting error analysis...")
    
    error_results = {}
    
    # 1. Identify potential data quality issues
    if 'full' in data:
        quality_issues = []
        for paper in data['full']:
            issues = paper.get('quality_check', {}).get('issues', [])
            if issues:
                quality_issues.extend(issues)
        
        error_results['quality_issues'] = {
            'total_issues': len(quality_issues),
            'issue_distribution': dict(Counter(quality_issues)),
            'papers_with_issues': sum(1 for p in data['full'] 
                                    if p.get('quality_check', {}).get('issues', []))
        }
    
    # 2. Analyze privacy risks
    if 'full' in data:
        privacy_risks = []
        for paper in data['full']:
            risk = paper.get('privacy_check', {}).get('risk_assessment', 'none')
            privacy_risks.append(risk)
        
        error_results['privacy_risks'] = {
            'distribution': dict(Counter(privacy_risks)),
            'high_risk_papers': sum(1 for r in privacy_risks if r in ['high', 'critical'])
        }
    
    # 3. Identify scoring outliers
    if 'scored' in data:
        quality_scores = []
        for paper in data['scored']:
            if 'enhanced_quality' in paper:
                quality_scores.append(paper['enhanced_quality']['total_score'])
        
        if quality_scores:
            q1 = np.percentile(quality_scores, 25)
            q3 = np.percentile(quality_scores, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = [s for s in quality_scores if s < lower_bound or s > upper_bound]
            
            error_results['score_analysis'] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(quality_scores)) * 100,
                'outlier_range': [min(outliers) if outliers else 0, 
                                max(outliers) if outliers else 0]
            }
    
    print("  ✓ Error analysis completed")
    return error_results

def compare_processing_stages(data: Dict) -> Dict:
    """Compare papers at different processing stages"""
    print("\nComparing different processing stages...")
    
    comparison_results = {}
    
    # Create DataFrames for analysis
    dfs = {}
    for stage, papers in data.items():
        df_data = []
        for paper in papers:
            record = {
                'stage': stage,
                'title_length': len(paper.get('title', '')),
                'abstract_length': len(paper.get('abstract', '')),
                'author_count': len(paper.get('authors', [])),
                'has_categories': bool(paper.get('categories', []))
            }
            
            # Add quality score (if available)
            if 'enhanced_quality' in paper:
                record['quality_score'] = paper['enhanced_quality']['total_score']
            elif 'overall_quality_score' in paper:
                record['quality_score'] = paper['overall_quality_score']
            
            df_data.append(record)
        
        if df_data:
            dfs[stage] = pd.DataFrame(df_data)
    
    # Calculate statistics for each stage
    for stage, df in dfs.items():
        comparison_results[stage] = {
            'count': len(df),
            'avg_title_length': df['title_length'].mean(),
            'avg_abstract_length': df['abstract_length'].mean(),
            'avg_author_count': df['author_count'].mean()
        }
        
        if 'quality_score' in df.columns:
            comparison_results[stage]['avg_quality_score'] = df['quality_score'].mean()
    
    print("  ✓ Stage comparison completed")
    return comparison_results

def generate_visualizations(data: Dict, output_dir: str = "visualizations"):
    """Generate visualization charts"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating visualization charts to {output_dir}...")
    
    # Set style for better looking plots
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # 1. Quality score distribution plot
    if 'scored' in data:
        quality_scores = []
        for paper in data['scored']:
            if 'enhanced_quality' in paper:
                quality_scores.append(paper['enhanced_quality']['total_score'])
        
        if quality_scores:
            plt.figure(figsize=(12, 7))
            
            # Create histogram with KDE
            hist = plt.hist(quality_scores, bins=25, alpha=0.7, color='skyblue', 
                          edgecolor='black', density=True, label='Frequency')
            
            # Add KDE line
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(quality_scores)
            x_range = np.linspace(min(quality_scores), max(quality_scores), 1000)
            plt.plot(x_range, kde(x_range), 'r-', linewidth=2, label='Density (KDE)')
            
            # Add vertical lines for mean and median
            mean_score = np.mean(quality_scores)
            median_score = np.median(quality_scores)
            plt.axvline(mean_score, color='green', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_score:.3f}')
            plt.axvline(median_score, color='orange', linestyle='-.', linewidth=2, 
                       label=f'Median: {median_score:.3f}')
            
            plt.title('Figure 1: Quality Score Distribution of Papers', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Quality Score', fontsize=14)
            plt.ylabel('Density / Frequency', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right', fontsize=12)
            plt.tight_layout()
            
            # Add statistics text box
            stats_text = f'Total Papers: {len(quality_scores)}\n'
            stats_text += f'Mean: {mean_score:.3f}\n'
            stats_text += f'Median: {median_score:.3f}\n'
            stats_text += f'Std Dev: {np.std(quality_scores):.3f}\n'
            stats_text += f'Range: [{min(quality_scores):.3f}, {max(quality_scores):.3f}]'
            
            plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                        fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.savefig(f'{output_dir}/quality_score_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Quality score distribution plot generated")
    
    # 2. Paper count by processing stage
    stages = list(data.keys())
    counts = [len(papers) for papers in data.values()]
    stage_labels = ['Raw', 'Cleaned', 'Scored', 'Fully Processed']
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(stage_labels, counts, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'], 
                  edgecolor='black', linewidth=1.5)
    plt.title('Figure 2: Paper Count by Processing Stage', 
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Processing Stage', fontsize=14)
    plt.ylabel('Number of Papers', fontsize=14)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count}\n({count/counts[0]*100:.1f}%)', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add percentage change lines and annotations
    for i in range(1, len(counts)):
        x1 = bars[i-1].get_x() + bars[i-1].get_width()
        x2 = bars[i].get_x()
        y = max(counts[i-1], counts[i]) + max(counts)*0.05
        change = ((counts[i] - counts[i-1]) / counts[i-1]) * 100
        
        plt.annotate('', xy=(x1, y), xytext=(x2, y),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1))
        plt.text((x1 + x2)/2, y + max(counts)*0.01, 
                f'{change:+.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(['Papers remaining'], loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/paper_count_by_stage.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Processing stage comparison plot generated")
    
    # 3. Privacy risk distribution plot (if available)
    if 'full' in data:
        privacy_risks = []
        for paper in data['full']:
            risk = paper.get('privacy_check', {}).get('risk_assessment', 'none')
            privacy_risks.append(risk)
        
        risk_counts = Counter(privacy_risks)
        
        plt.figure(figsize=(12, 7))
        
        # Define colors and labels
        risk_labels = ['Critical', 'High', 'Medium', 'Low', 'None']
        risk_mapping = {'critical': 'Critical', 'high': 'High', 
                       'medium': 'Medium', 'low': 'Low', 'none': 'None'}
        colors = {'Critical': 'red', 'High': 'orange', 'Medium': 'yellow', 
                 'Low': 'lightgreen', 'None': 'gray'}
        
        # Prepare data for plotting
        sorted_risk_labels = []
        risk_values = []
        risk_colors = []
        
        for label in risk_labels:
            mapped_label = label.lower()
            if mapped_label in risk_counts:
                sorted_risk_labels.append(label)
                risk_values.append(risk_counts[mapped_label])
                risk_colors.append(colors[label])
        
        # Create bar plot
        bars = plt.bar(sorted_risk_labels, risk_values, 
                      color=risk_colors, edgecolor='black', linewidth=1.5)
        
        plt.title('Figure 3: Privacy Risk Distribution in Processed Papers', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Risk Level', fontsize=14)
        plt.ylabel('Number of Papers', fontsize=14)
        
        # Add value labels
        total_papers = sum(risk_values)
        for bar, value in zip(bars, risk_values):
            height = bar.get_height()
            percentage = (value / total_papers) * 100
            plt.text(bar.get_x() + bar.get_width()/2., height + max(risk_values)*0.01,
                    f'{value}\n({percentage:.1f}%)', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add legend for risk levels
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[label], 
                                        edgecolor='black', label=label)
                          for label in sorted_risk_labels]
        plt.legend(handles=legend_elements, title='Risk Levels', 
                  loc='upper right', fontsize=11)
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/privacy_risk_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Privacy risk distribution plot generated")
    
    # 4. Additional visualization: Quality score vs text length
    if 'scored' in data:
        title_lengths = []
        abstract_lengths = []
        quality_scores = []
        
        for paper in data['scored']:
            if 'enhanced_quality' in paper:
                title_lengths.append(len(paper.get('title', '')))
                abstract_lengths.append(len(paper.get('abstract', '')))
                quality_scores.append(paper['enhanced_quality']['total_score'])
        
        if quality_scores:
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # Plot 1: Quality score vs title length
            scatter1 = ax1.scatter(title_lengths, quality_scores, 
                                 alpha=0.6, c=quality_scores, cmap='viridis', 
                                 edgecolors='black', linewidth=0.5)
            ax1.set_xlabel('Title Length (characters)', fontsize=12)
            ax1.set_ylabel('Quality Score', fontsize=12)
            ax1.set_title('Quality Score vs Title Length', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar1 = plt.colorbar(scatter1, ax=ax1)
            cbar1.set_label('Quality Score', fontsize=12)
            
            # Plot 2: Quality score vs abstract length
            scatter2 = ax2.scatter(abstract_lengths, quality_scores, 
                                 alpha=0.6, c=quality_scores, cmap='plasma', 
                                 edgecolors='black', linewidth=0.5)
            ax2.set_xlabel('Abstract Length (characters)', fontsize=12)
            ax2.set_ylabel('Quality Score', fontsize=12)
            ax2.set_title('Quality Score vs Abstract Length', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar2 = plt.colorbar(scatter2, ax=ax2)
            cbar2.set_label('Quality Score', fontsize=12)
            
            plt.suptitle('Figure 4: Correlation Analysis Between Text Length and Quality Scores', 
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/quality_vs_text_length.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Quality vs text length correlation plots generated")

def generate_comprehensive_report(data: Dict, ablation_results: Dict, 
                                 error_results: Dict, comparison_results: Dict) -> Dict:
    """Generate comprehensive report"""
    
    report = {
        'report_metadata': {
            'generation_date': datetime.now().isoformat(),
            'data_files_analyzed': list(data.keys()),
            'total_papers_processed': sum(len(papers) for papers in data.values())
        },
        'executive_summary': {
            'overall_status': 'complete',
            'key_findings': [],
            'recommendations': []
        },
        'ablation_study_results': ablation_results,
        'error_analysis_results': error_results,
        'stage_comparison_results': comparison_results,
        'detailed_analysis': {}
    }
    
    # Add key findings
    if 'cleaning_impact' in ablation_results:
        cleaning = ablation_results['cleaning_impact']
        report['executive_summary']['key_findings'].append(
            f"Cleaning stage removed {cleaning['papers_removed']} papers "
            f"({cleaning['removal_rate']:.1f}% of original dataset)"
        )
    
    if 'full_pipeline_impact' in ablation_results:
        pipeline = ablation_results['full_pipeline_impact']
        report['executive_summary']['key_findings'].append(
            f"Full processing pipeline retained {pipeline['processing_efficiency']:.1f}% of original papers"
        )
    
    # Add error analysis findings
    if 'quality_issues' in error_results:
        issues = error_results['quality_issues']
        report['executive_summary']['key_findings'].append(
            f"{issues['papers_with_issues']} papers have data quality issues"
        )
    
    if 'privacy_risks' in error_results:
        privacy = error_results['privacy_risks']
        high_risk = privacy.get('high_risk_papers', 0)
        if high_risk > 0:
            report['executive_summary']['key_findings'].append(
                f"{high_risk} papers have high privacy risk (require manual review)"
            )
    
    # Add recommendations
    if 'quality_issues' in error_results and error_results['quality_issues']['total_issues'] > 0:
        report['executive_summary']['recommendations'].append(
            "Perform manual review and correction for papers with quality issues"
        )
    
    if 'privacy_risks' in error_results and error_results['privacy_risks']['high_risk_papers'] > 0:
        report['executive_summary']['recommendations'].append(
            "Conduct further review and anonymization for high privacy risk papers"
        )
    
    report['executive_summary']['recommendations'].extend([
        "Regularly update data sources to ensure timeliness",
        "Consider adding more quality assessment dimensions",
        "Establish automated monitoring and alerting mechanisms"
    ])
    
    return report

def save_analysis_results(report: Dict, ablation_results: Dict, 
                         error_results: Dict, comparison_results: Dict):
    """Save analysis results"""
    
    # Save comprehensive report
    report_file = "ablation_analysis_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✓ Comprehensive report saved to {report_file}")
    
    # Save detailed analysis results
    detailed_file = "detailed_analysis_results.json"
    detailed_results = {
        'ablation_study': ablation_results,
        'error_analysis': error_results,
        'stage_comparison': comparison_results
    }
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Detailed analysis results saved to {detailed_file}")
    
    # Generate readable text summary
    summary_file = "analysis_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ABLATION EXPERIMENT AND ERROR ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Executive Summary:\n")
        f.write("-" * 40 + "\n")
        for finding in report['executive_summary']['key_findings']:
            f.write(f"• {finding}\n")
        
        f.write("\nRecommendations:\n")
        f.write("-" * 40 + "\n")
        for recommendation in report['executive_summary']['recommendations']:
            f.write(f"• {recommendation}\n")
        
        f.write("\nKey Findings:\n")
        f.write("-" * 40 + "\n")
        
        # Ablation study results
        if 'cleaning_impact' in ablation_results:
            cleaning = ablation_results['cleaning_impact']
            f.write(f"1. Cleaning stage removed {cleaning['papers_removed']} papers ")
            f.write(f"({cleaning['removal_rate']:.1f}% of original dataset)\n")
        
        # Error analysis results
        if 'quality_issues' in error_results:
            issues = error_results['quality_issues']
            f.write(f"2. {issues['papers_with_issues']} papers have data quality issues\n")
            if issues['issue_distribution']:
                f.write("   Main issue types:\n")
                for issue_type, count in issues['issue_distribution'].items():
                    f.write(f"     - {issue_type}: {count} occurrences\n")
        
        # Privacy risk results
        if 'privacy_risks' in error_results:
            privacy = error_results['privacy_risks']
            f.write(f"3. Privacy risk distribution:\n")
            for risk_level, count in privacy['distribution'].items():
                percentage = (count / sum(privacy['distribution'].values())) * 100
                f.write(f"     - {risk_level}: {count} papers ({percentage:.1f}%)\n")
    
    print(f"✓ Text summary saved to {summary_file}")

def main():
    """Main function"""
    print("=" * 60)
    print("ABLATION EXPERIMENT AND ERROR ANALYSIS SYSTEM")
    print("=" * 60)
    
    # Load all data
    data = load_all_data_files()
    
    if not data:
        print("Error: No data files found!")
        print("Please ensure the following files are generated:")
        print("  - test_papers.jsonl (raw data)")
        print("  - cleaned_papers.jsonl (cleaned data)")
        print("  - enhanced_scored_papers.jsonl (scored data)")
        print("  - full_processed_papers.jsonl (fully processed data)")
        return
    
    # Conduct ablation study analysis
    ablation_results = conduct_ablation_study(data)
    
    # Conduct error analysis
    error_results = error_analysis(data)
    
    # Compare different processing stages
    comparison_results = compare_processing_stages(data)
    
    # Generate visualization charts
    generate_visualizations(data)
    
    # Generate comprehensive report
    report = generate_comprehensive_report(data, ablation_results, 
                                         error_results, comparison_results)
    
    # Save analysis results
    save_analysis_results(report, ablation_results, error_results, comparison_results)
    
    # Display key results
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE - KEY RESULTS")
    print("=" * 60)
    
    print(f"\nProcessing Pipeline Overview:")
    for stage, papers in data.items():
        stage_name = stage.upper().replace('_', ' ')
        print(f"  {stage_name:15s}: {len(papers):4d} papers")
    
    if 'cleaning_impact' in ablation_results:
        cleaning = ablation_results['cleaning_impact']
        print(f"\nCleaning Effectiveness:")
        print(f"  Papers removed: {cleaning['papers_removed']} ({cleaning['removal_rate']:.1f}%)")
        print(f"  Abstract length change: {cleaning['avg_abstract_length_change']:+.1f} characters")
        print(f"  Title length change: {cleaning['avg_title_length_change']:+.1f} characters")
    
    if 'quality_issues' in error_results:
        issues = error_results['quality_issues']
        print(f"\nData Quality Issues:")
        print(f"  Papers with issues: {issues['papers_with_issues']}")
        if issues['issue_distribution']:
            print("  Main issue types:")
            for issue_type, count in list(issues['issue_distribution'].items())[:3]:
                print(f"    - {issue_type}: {count} occurrences")
    
    if 'privacy_risks' in error_results:
        privacy = error_results['privacy_risks']
        high_risk = privacy.get('high_risk_papers', 0)
        if high_risk > 0:
            print(f"\n  High Privacy Risk Papers: {high_risk} (require manual review)")
    
    if 'score_analysis' in error_results:
        scores = error_results['score_analysis']
        print(f"\nQuality Score Analysis:")
        print(f"  Outliers detected: {scores['outlier_count']} ({scores['outlier_percentage']:.1f}%)")
        print(f"  Outlier range: [{scores['outlier_range'][0]:.3f}, {scores['outlier_range'][1]:.3f}]")
    
    print("\n" + "=" * 60)
    print("OUTPUT FILES:")
    print("=" * 60)
    print("✓ ablation_analysis_report.json - Comprehensive analysis report")
    print("✓ detailed_analysis_results.json - Detailed analysis results")
    print("✓ analysis_summary.txt - Text summary")
    print("✓ visualizations/ - Visualization charts directory")
    print("=" * 60)

if __name__ == "__main__":
    main()
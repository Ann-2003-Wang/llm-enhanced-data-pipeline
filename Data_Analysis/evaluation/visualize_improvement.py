# visualize_improvement.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams

# Set style for better visuals
plt.style.use('seaborn-v0_8-darkgrid')
rcParams.update({
    'figure.autolayout': True,
    'figure.figsize': (12, 7),
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

# Read data from CSV file
def load_and_process_data(csv_file):
    """Load and preprocess the data from CSV file"""
    df = pd.read_csv(csv_file)
    
    # Extract numeric values from score columns (e.g., convert "5.21/10" to 5.21)
    score_columns = ['Avg_Novelty_Score', 'Avg_Technical_Depth', 
                     'Avg_Clarity_Score', 'Avg_Impact_Potential', 'Avg_Overall_Score']
    
    for col in score_columns:
        df[f'{col}_value'] = df[col].apply(lambda x: float(str(x).split('/')[0]) if isinstance(x, str) and '/' in x else float(x))
    
    return df

def create_grouped_bar_chart(df, output_file='paper_quality_comparison.png'):
    """Create a grouped bar chart comparing metrics across stages"""
    
    # Define stages and metrics
    stages = df['Stage'].tolist()
    metrics = ['Novelty', 'Technical Depth', 'Clarity', 'Impact Potential', 'Overall Score']
    metric_columns = ['Avg_Novelty_Score_value', 'Avg_Technical_Depth_value', 
                     'Avg_Clarity_Score_value', 'Avg_Impact_Potential_value', 
                     'Avg_Overall_Score_value']
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. Main grouped bar chart for quality metrics
    x = np.arange(len(stages))
    width = 0.15
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, (metric, col, color) in enumerate(zip(metrics, metric_columns, colors)):
        offset = (i - len(metrics)/2) * width + width/2
        values = df[col].values
        bars = ax1.bar(x + offset, values, width, label=metric, color=color, edgecolor='black', linewidth=0.5)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax1.set_xlabel('Processing Stage')
    ax1.set_ylabel('Average Score (out of 10)')
    ax1.set_title('Paper Quality Metrics Across Processing Stages', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{stage}\n(n={int(df.loc[df['Stage']==stage, 'Num_Papers'].values[0])})" 
                         for stage in stages], fontsize=10)
    ax1.set_ylim(0, 10.5)
    ax1.axhline(y=5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.legend(title='Quality Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)
    
    # 2. Confidence and paper count subplot
    fig.subplots_adjust(hspace=0.3)
    
    # Confidence line with markers
    confidence_values = df['Confidence'].values
    line1 = ax2.plot(x, confidence_values, marker='o', markersize=8, 
                     linewidth=2.5, color='#e74c3c', label='Confidence', zorder=3)
    ax2.set_xlabel('Processing Stage')
    ax2.set_ylabel('Confidence Score', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.set_ylim(0.6, 1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages, fontsize=10)
    ax2.set_title('Confidence Score and Paper Count Reduction', fontsize=14, fontweight='bold', pad=15)
    
    # Add confidence value labels
    for i, (xi, yi) in enumerate(zip(x, confidence_values)):
        ax2.text(xi, yi + 0.005, f'{yi:.3f}', ha='center', va='bottom', 
                fontsize=9, fontweight='bold', color='#e74c3c')
    
    # Paper count as bar chart on secondary y-axis
    ax2b = ax2.twinx()
    paper_counts = df['Num_Papers'].values
    colors_bar = ['#95a5a6', '#7f8c8d', '#34495e']
    bars = ax2b.bar(x, paper_counts, width=0.6, alpha=0.6, 
                    color=colors_bar, edgecolor='black', linewidth=0.5, label='Paper Count')
    ax2b.set_ylabel('Number of Papers', color='#2c3e50')
    ax2b.tick_params(axis='y', labelcolor='#2c3e50')
    
    # Add paper count labels
    for bar, count in zip(bars, paper_counts):
        height = bar.get_height()
        ax2b.text(bar.get_x() + bar.get_width()/2., height + 50,
                 f'{count:,}', ha='center', va='bottom', fontsize=9, 
                 fontweight='bold', color='#2c3e50')
    
    # Add legend for both lines
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Add grid
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)
    
    # Add percentage reduction text
    reduction_text = []
    for i in range(1, len(stages)):
        reduction = (1 - paper_counts[i]/paper_counts[i-1]) * 100
        reduction_text.append(f'{reduction:.1f}% reduction')
    
    if reduction_text:
        ax2.text(0.02, 0.98, f'Stage 1→2: {reduction_text[0]}\nStage 2→3: {reduction_text[1]}', 
                transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Add overall improvement annotation
    overall_improvement = ((df.loc[df['Stage']=='Enhanced & Filtered', 'Avg_Overall_Score_value'].values[0] - 
                           df.loc[df['Stage']=='Raw (Merged)', 'Avg_Overall_Score_value'].values[0]) / 
                           df.loc[df['Stage']=='Raw (Merged)', 'Avg_Overall_Score_value'].values[0]) * 100
    
    ax1.annotate(f'Overall Score Improvement: {overall_improvement:.1f}%', 
                xy=(2, df.loc[df['Stage']=='Enhanced & Filtered', 'Avg_Overall_Score_value'].values[0]), 
                xytext=(1.5, 9.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                fontsize=10, fontweight='bold', color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_radar_chart(df, output_file='paper_quality_radar.png'):
    """Create a radar chart showing metrics for each stage"""
    
    # Extract data
    stages = df['Stage'].tolist()
    metrics = ['Novelty', 'Technical Depth', 'Clarity', 'Impact Potential', 'Overall Score']
    metric_columns = ['Avg_Novelty_Score_value', 'Avg_Technical_Depth_value', 
                     'Avg_Clarity_Score_value', 'Avg_Impact_Potential_value', 
                     'Avg_Overall_Score_value']
    
    # Prepare data for radar chart
    data = {}
    for i, stage in enumerate(stages):
        data[stage] = df.loc[df['Stage']==stage, metric_columns].values[0]
    
    # Number of variables
    N = len(metrics)
    
    # Create angles for radar chart
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # Colors for each stage
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # Plot each stage
    for i, (stage, color) in enumerate(zip(stages, colors)):
        values = data[stage].tolist()
        values += values[:1]  # Close the polygon
        ax.plot(angles, values, 'o-', linewidth=2, label=stage, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9)
    ax.set_title('Paper Quality Metrics Radar Chart', fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_improvement_bar_chart(df, output_file='quality_improvement.png'):
    """Create a bar chart showing improvement percentages"""
    
    # Calculate percentage improvement from Raw to Enhanced stage
    metrics = ['Novelty', 'Technical Depth', 'Clarity', 'Impact Potential', 'Overall Score']
    metric_columns = ['Avg_Novelty_Score_value', 'Avg_Technical_Depth_value', 
                     'Avg_Clarity_Score_value', 'Avg_Impact_Potential_value', 
                     'Avg_Overall_Score_value']
    
    raw_values = df.loc[df['Stage']=='Raw (Merged)', metric_columns].values[0]
    enhanced_values = df.loc[df['Stage']=='Enhanced & Filtered', metric_columns].values[0]
    
    improvements = ((enhanced_values - raw_values) / raw_values) * 100
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(metrics, improvements, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])
    ax.set_xlabel('Quality Metric')
    ax.set_ylabel('Improvement Percentage (%)')
    ax.set_title('Percentage Improvement from Raw to Enhanced Stage', fontsize=16, fontweight='bold', pad=20)
    
    # Add value labels
    for bar, value in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """Main function to generate all visualizations"""
    
    # Create sample CSV data (replace with actual file reading)
    data = """Stage,Num_Papers,Avg_Novelty_Score,Avg_Technical_Depth,Avg_Clarity_Score,Avg_Impact_Potential,Avg_Overall_Score,Confidence
Raw (Merged),7397,5.21/10,5.68/10,6.94/10,5.89/10,5.93/10,0.742
Cleaned & Aligned,6242,5.47/10,5.91/10,7.56/10,6.32/10,6.28/10,0.801
Enhanced & Filtered,3236,5.93/10,6.33/10,8.02/10,6.96/10,6.81/10,0.851"""
    
    # Save sample data to CSV
    with open('paper_quality_data.csv', 'w') as f:
        f.write(data)
    
    print("Sample CSV file created: paper_quality_data.csv")
    
    # Load and process data
    try:
        df = load_and_process_data('paper_quality_data.csv')
        print("\nData loaded successfully:")
        print(df.to_string())
        
        # Create visualizations
        print("\nCreating visualizations...")
        
        # 1. Main grouped bar chart
        print("1. Creating grouped bar chart...")
        fig1 = create_grouped_bar_chart(df, 'paper_quality_comparison.png')
        
        # 2. Radar chart
        print("2. Creating radar chart...")
        fig2 = create_radar_chart(df, 'paper_quality_radar.png')
        
        # 3. Improvement chart
        print("3. Creating improvement chart...")
        fig3 = create_improvement_bar_chart(df, 'quality_improvement.png')
        
        print("\nVisualizations saved as:")
        print("  - paper_quality_comparison.png")
        print("  - paper_quality_radar.png")
        print("  - quality_improvement.png")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("-" * 50)
        print(f"Total papers processed: {df['Num_Papers'].sum():,}")
        print(f"Final papers retained: {df.loc[df['Stage']=='Enhanced & Filtered', 'Num_Papers'].values[0]:,}")
        retention_rate = (df.loc[df['Stage']=='Enhanced & Filtered', 'Num_Papers'].values[0] / 
                        df.loc[df['Stage']=='Raw (Merged)', 'Num_Papers'].values[0]) * 100
        print(f"Retention rate: {retention_rate:.1f}%")
        print(f"Overall quality improvement: {((df.loc[df['Stage']=='Enhanced & Filtered', 'Avg_Overall_Score_value'].values[0] - df.loc[df['Stage']=='Raw (Merged)', 'Avg_Overall_Score_value'].values[0]) / df.loc[df['Stage']=='Raw (Merged)', 'Avg_Overall_Score_value'].values[0]) * 100:.1f}%")
        print(f"Confidence improvement: {((df.loc[df['Stage']=='Enhanced & Filtered', 'Confidence'].values[0] - df.loc[df['Stage']=='Raw (Merged)', 'Confidence'].values[0]) / df.loc[df['Stage']=='Raw (Merged)', 'Confidence'].values[0]) * 100:.1f}%")
        
    except FileNotFoundError:
        print("Error: CSV file not found. Please ensure 'paper_quality_data.csv' exists in the current directory.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
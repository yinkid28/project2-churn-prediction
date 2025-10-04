"""
Customer Lifetime Value (CLV) Analysis
======================================
Analyzes CLV distribution and relationship with churn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_processed_data():
    """Load the processed training data."""
    
    train_df = pd.read_csv('data/processed/train.csv')
    encoders = joblib.load('models/encoders.pkl')
    
    return train_df, encoders

def decode_clv_quartile(quartile_encoded, encoders):
    """Decode CLV quartile back to readable labels."""
    
    if 'clv_quartile' in encoders:
        quartile_decoder = {i: label for i, label in enumerate(encoders['clv_quartile'].classes_)}
        return quartile_decoder.get(quartile_encoded, 'Unknown')
    return quartile_encoded

def analyze_clv_distribution(df, encoders):
    """Analyze CLV distribution and create visualizations."""
    
    print("=== CLV Distribution Analysis ===\n")
    
    # Basic CLV statistics
    print("CLV Statistics:")
    print(df['clv'].describe())
    print()
    
    # CLV by quartile
    print("CLV by Quartile:")
    clv_by_quartile = df.groupby('clv_quartile')['clv'].agg(['count', 'mean', 'std'])
    
    # Decode quartile labels if they're encoded
    if 'clv_quartile' in encoders:
        quartile_labels = [encoders['clv_quartile'].classes_[i] for i in clv_by_quartile.index]
        clv_by_quartile.index = quartile_labels
    
    print(clv_by_quartile)
    print()
    
    return clv_by_quartile

def analyze_churn_by_clv(df, encoders):
    """Analyze churn rate by CLV quartile."""
    
    print("=== Churn Rate by CLV Quartile ===\n")
    
    # Calculate churn rate by quartile
    churn_by_clv = df.groupby('clv_quartile')['Churn'].agg(['count', 'sum', 'mean']).round(3)
    churn_by_clv.columns = ['Total_Customers', 'Churned', 'Churn_Rate']
    
    # Decode quartile labels if they're encoded
    if 'clv_quartile' in encoders:
        quartile_labels = [encoders['clv_quartile'].classes_[i] for i in churn_by_clv.index]
        churn_by_clv.index = quartile_labels
    
    print("Churn Analysis by CLV Quartile:")
    print(churn_by_clv)
    print()
    
    # Calculate expected revenue loss by quartile
    revenue_analysis = df.groupby('clv_quartile').agg({
        'clv': 'mean',
        'Churn': 'sum',
        'MonthlyCharges': 'mean'
    }).round(2)
    
    if 'clv_quartile' in encoders:
        revenue_analysis.index = quartile_labels
    
    revenue_analysis['Revenue_at_Risk'] = (
        revenue_analysis['clv'] * revenue_analysis['Churn']
    ).round(2)
    
    print("Revenue Analysis by CLV Quartile:")
    print(revenue_analysis)
    print()
    
    return churn_by_clv, revenue_analysis

def create_visualizations(df, encoders):
    """Create CLV and churn analysis visualizations."""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Customer Lifetime Value & Churn Analysis', fontsize=16, fontweight='bold')
    
    # 1. CLV Distribution
    axes[0, 0].hist(df['clv'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('CLV Distribution')
    axes[0, 0].set_xlabel('Customer Lifetime Value ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. CLV by Quartile
    clv_quartile_means = df.groupby('clv_quartile')['clv'].mean()
    
    # Decode quartile labels for plotting
    if 'clv_quartile' in encoders:
        quartile_labels = [encoders['clv_quartile'].classes_[i] for i in clv_quartile_means.index]
        clv_quartile_means.index = quartile_labels
    
    bars1 = axes[0, 1].bar(clv_quartile_means.index, clv_quartile_means.values, 
                          color='lightcoral', alpha=0.8, edgecolor='black')
    axes[0, 1].set_title('Average CLV by Quartile')
    axes[0, 1].set_xlabel('CLV Quartile')
    axes[0, 1].set_ylabel('Average CLV ($)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 50,
                       f'${height:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Churn Rate by CLV Quartile
    churn_rates = df.groupby('clv_quartile')['Churn'].mean()
    
    if 'clv_quartile' in encoders:
        churn_rates.index = quartile_labels
    
    bars2 = axes[1, 0].bar(churn_rates.index, churn_rates.values * 100, 
                          color='orange', alpha=0.8, edgecolor='black')
    axes[1, 0].set_title('Churn Rate by CLV Quartile')
    axes[1, 0].set_xlabel('CLV Quartile')
    axes[1, 0].set_ylabel('Churn Rate (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar in bars2:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Revenue at Risk by Quartile
    revenue_at_risk = df.groupby('clv_quartile').agg({
        'clv': 'mean',
        'Churn': 'sum'
    })
    revenue_at_risk['risk'] = revenue_at_risk['clv'] * revenue_at_risk['Churn']
    
    if 'clv_quartile' in encoders:
        revenue_at_risk.index = quartile_labels
    
    bars3 = axes[1, 1].bar(revenue_at_risk.index, revenue_at_risk['risk'], 
                          color='red', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Total Revenue at Risk by Quartile')
    axes[1, 1].set_xlabel('CLV Quartile')
    axes[1, 1].set_ylabel('Revenue at Risk ($)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 500,
                       f'${height:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('clv_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def generate_business_insights(df, encoders):
    """Generate key business insights from CLV analysis."""
    
    print("=== KEY BUSINESS INSIGHTS ===\n")
    
    # Calculate key metrics
    churn_by_clv = df.groupby('clv_quartile')['Churn'].mean()
    clv_by_quartile = df.groupby('clv_quartile')['clv'].mean()
    
    # Decode quartile labels
    if 'clv_quartile' in encoders:
        quartile_labels = [encoders['clv_quartile'].classes_[i] for i in churn_by_clv.index]
        churn_by_clv.index = quartile_labels
        clv_by_quartile.index = quartile_labels
    
    insights = []
    
    # Insight 1: CLV vs Churn relationship
    highest_clv_quartile = clv_by_quartile.idxmax()
    highest_clv_churn = churn_by_clv[highest_clv_quartile]
    lowest_clv_quartile = clv_by_quartile.idxmin()
    lowest_clv_churn = churn_by_clv[lowest_clv_quartile]
    
    insights.append(
        f"**Retention Priority**: {highest_clv_quartile} CLV customers have a "
        f"{highest_clv_churn:.1%} churn rate despite being most valuable "
        f"(avg CLV: ${clv_by_quartile[highest_clv_quartile]:.0f}). These should be "
        f"top retention priority."
    )
    
    # Insight 2: Revenue at Risk
    revenue_at_risk = df.groupby('clv_quartile').agg({
        'clv': 'mean',
        'Churn': 'sum'
    })
    revenue_at_risk['total_risk'] = revenue_at_risk['clv'] * revenue_at_risk['Churn']
    
    if 'clv_quartile' in encoders:
        revenue_at_risk.index = quartile_labels
    
    highest_risk_quartile = revenue_at_risk['total_risk'].idxmax()
    highest_risk_amount = revenue_at_risk['total_risk'].max()
    
    insights.append(
        f"**Revenue Risk**: {highest_risk_quartile} CLV segment has the highest "
        f"total revenue at risk (${highest_risk_amount:.0f}), making it the most "
        f"critical segment for retention campaigns."
    )
    
    # Insight 3: Efficiency of retention efforts
    if highest_clv_churn < lowest_clv_churn:
        insights.append(
            f"**Retention Efficiency**: Counter-intuitively, higher CLV customers "
            f"actually churn less ({highest_clv_churn:.1%} vs {lowest_clv_churn:.1%}), "
            f"suggesting retention efforts should focus on understanding why lower "
            f"CLV customers leave more frequently."
        )
    else:
        insights.append(
            f"**Retention Challenge**: Higher CLV customers churn more frequently "
            f"({highest_clv_churn:.1%} vs {lowest_clv_churn:.1%}), indicating urgent "
            f"need for premium customer retention strategies."
        )
    
    # Print insights
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}\n")
    
    return insights

def main():
    """Main CLV analysis pipeline."""
    
    print("=== Customer Lifetime Value Analysis ===\n")
    
    try:
        # Load processed data
        print("Loading processed data...")
        df, encoders = load_processed_data()
        
        # Analyze CLV distribution
        clv_stats = analyze_clv_distribution(df, encoders)
        
        # Analyze churn by CLV
        churn_analysis, revenue_analysis = analyze_churn_by_clv(df, encoders)
        
        # Create visualizations
        print("Creating visualizations...")
        fig = create_visualizations(df, encoders)
        
        # Generate business insights
        insights = generate_business_insights(df, encoders)
        
        print("✅ CLV analysis complete!")
        print("📊 Visualization saved as 'clv_analysis.png'")
        
        return {
            'clv_stats': clv_stats,
            'churn_analysis': churn_analysis,
            'revenue_analysis': revenue_analysis,
            'insights': insights
        }
        
    except FileNotFoundError:
        print("❌ Error: Processed data not found. Please run data preparation first.")
        return None

if __name__ == "__main__":
    results = main()
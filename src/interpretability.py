"""
Model Interpretability - SHAP Analysis and Feature Importance
============================================================
Provides model explanations using SHAP for tree-based models and 
coefficient analysis for logistic regression.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP, fallback to feature importance if not available
try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP library available - using TreeExplainer for tree models")
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available - using fallback feature importance methods")

def load_models_and_data():
    """Load trained models and test data."""
    
    # Load models
    models = {
        'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
        'Random Forest': joblib.load('models/random_forest.pkl'),
        'XGBoost': joblib.load('models/xgboost.pkl')
    }
    
    # Load test data
    test_df = pd.read_csv('data/processed/test.csv')
    feature_cols = joblib.load('models/feature_columns.pkl')
    
    X_test = test_df[feature_cols]
    y_test = test_df['Churn']
    
    return models, X_test, y_test, feature_cols

def get_logistic_feature_importance(model, feature_names, X_sample):
    """
    Calculate feature importance for logistic regression using 
    standardized coefficients.
    """
    
    # Get the logistic regression model from pipeline
    if hasattr(model, 'named_steps'):
        lr_model = model.named_steps['classifier']
        scaler = model.named_steps['scaler']
        
        # Get standardized coefficients
        coefficients = lr_model.coef_[0]
        
        # Calculate feature standard deviations from scaled data
        X_scaled = scaler.transform(X_sample)
        feature_stds = np.std(X_scaled, axis=0)
        
        # Calculate importance as |coefficient * std_dev_of_feature|
        importance = np.abs(coefficients * feature_stds)
        
    else:
        # If not a pipeline, assume it's a direct model
        coefficients = model.coef_[0]
        feature_stds = np.std(X_sample, axis=0)
        importance = np.abs(coefficients * feature_stds)
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
        'coefficient': coefficients if hasattr(model, 'named_steps') else model.coef_[0]
    }).sort_values('importance', ascending=False)
    
    return importance_df

def get_tree_feature_importance(model, feature_names):
    """Get feature importance for tree-based models."""
    
    # Get the tree model from pipeline
    if hasattr(model, 'named_steps'):
        tree_model = model.named_steps['classifier']
    else:
        tree_model = model
    
    importance = tree_model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df

def create_shap_explainer(model, X_sample, model_name):
    """Create SHAP explainer for the given model."""
    
    if not SHAP_AVAILABLE:
        return None
    
    try:
        if 'Logistic' in model_name:
            # For logistic regression, we'll skip SHAP as coefficients are more interpretable
            print(f"Skipping SHAP for {model_name} - using coefficient analysis instead")
            return None
            
        elif any(name in model_name for name in ['Random Forest', 'XGBoost']):
            # Get the actual model from pipeline
            if hasattr(model, 'named_steps'):
                tree_model = model.named_steps['classifier']
            else:
                tree_model = model
            
            # Create TreeExplainer
            explainer = shap.TreeExplainer(tree_model)
            
            # Test with a small sample to ensure it works
            shap_values = explainer.shap_values(X_sample.iloc[:5])
            
            print(f"SHAP TreeExplainer created successfully for {model_name}")
            return explainer
            
    except Exception as e:
        print(f"Failed to create SHAP explainer for {model_name}: {e}")
        return None
    
    return None

def get_global_feature_importance(models, X_test, feature_names, top_n=10):
    """Get global feature importance for all models."""
    
    print("=== GLOBAL FEATURE IMPORTANCE ===\n")
    
    all_importance = {}
    
    for model_name, model in models.items():
        print(f"Analyzing {model_name}...")
        
        if 'Logistic' in model_name:
            # Use coefficient analysis for logistic regression
            importance_df = get_logistic_feature_importance(model, feature_names, X_test)
            all_importance[model_name] = importance_df.head(top_n)
            
            print(f"Top {top_n} features (by |coefficient * std|):")
            for idx, row in importance_df.head(top_n).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f} (coef: {row['coefficient']:.4f})")
                
        else:
            # Use built-in feature importance for tree models
            importance_df = get_tree_feature_importance(model, feature_names)
            all_importance[model_name] = importance_df.head(top_n)
            
            print(f"Top {top_n} features (by importance):")
            for idx, row in importance_df.head(top_n).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        print()
    
    return all_importance

def create_global_importance_plot(all_importance):
    """Create visualization of global feature importance."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Global Feature Importance by Model', fontsize=16, fontweight='bold')
    
    for idx, (model_name, importance_df) in enumerate(all_importance.items()):
        ax = axes[idx]
        
        # Plot top 10 features
        top_features = importance_df.head(10)
        
        bars = ax.barh(range(len(top_features)), top_features['importance'], 
                      color=['skyblue', 'lightcoral', 'lightgreen'][idx], alpha=0.8)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'{model_name}')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('global_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def get_shap_global_importance(models, X_test, feature_names, sample_size=200):
    """Get SHAP global importance if available."""
    
    if not SHAP_AVAILABLE:
        print("SHAP not available - skipping SHAP global analysis")
        return None
    
    print("=== SHAP GLOBAL IMPORTANCE ===\n")
    
    shap_results = {}
    
    # Use a sample for faster computation
    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    
    for model_name, model in models.items():
        if 'Logistic' in model_name:
            continue  # Skip logistic regression for SHAP
            
        print(f"Computing SHAP values for {model_name}...")
        
        explainer = create_shap_explainer(model, X_sample, model_name)
        if explainer is None:
            continue
            
        try:
            # Get SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # For binary classification, shap_values might be a list
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
            
            # Create summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", 
                            feature_names=feature_names, show=False)
            plt.title(f'SHAP Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig(f'shap_importance_{model_name.lower().replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            # Store results
            feature_importance = np.abs(shap_values).mean(axis=0)
            shap_results[model_name] = pd.DataFrame({
                'feature': feature_names,
                'shap_importance': feature_importance
            }).sort_values('shap_importance', ascending=False)
            
            print(f"✅ SHAP analysis completed for {model_name}")
            
        except Exception as e:
            print(f"❌ SHAP analysis failed for {model_name}: {e}")
            continue
    
    return shap_results

def explain_single_prediction(models, customer_data, feature_names, customer_id="SAMPLE"):
    """Provide local explanation for a single customer prediction."""
    
    print(f"=== LOCAL EXPLANATION FOR CUSTOMER {customer_id} ===\n")
    
    explanations = {}
    
    for model_name, model in models.items():
        print(f"Explaining {model_name} prediction...")
        
        # Get prediction
        prob = model.predict_proba(customer_data)[0, 1]
        print(f"Churn probability: {prob:.3f}")
        
        if 'Logistic' in model_name:
            # For logistic regression, show top contributing features
            if hasattr(model, 'named_steps'):
                lr_model = model.named_steps['classifier']
                scaler = model.named_steps['scaler']
                customer_scaled = scaler.transform(customer_data)
            else:
                lr_model = model
                customer_scaled = customer_data.values
            
            # Calculate contribution: coefficient * feature_value
            contributions = lr_model.coef_[0] * customer_scaled[0]
            
            contrib_df = pd.DataFrame({
                'feature': feature_names,
                'contribution': contributions,
                'feature_value': customer_data.iloc[0].values
            }).sort_values('contribution', key=abs, ascending=False)
            
            print("Top contributing features:")
            for idx, row in contrib_df.head(5).iterrows():
                direction = "increases" if row['contribution'] > 0 else "decreases"
                print(f"  {row['feature']}: {row['contribution']:.4f} ({direction} churn risk)")
            
            explanations[model_name] = contrib_df.head(10)
        
        else:
            # Try SHAP for tree models
            if SHAP_AVAILABLE:
                explainer = create_shap_explainer(model, customer_data, model_name)
                if explainer is not None:
                    try:
                        shap_values = explainer.shap_values(customer_data)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]  # Positive class
                        
                        # Create explanation dataframe
                        contrib_df = pd.DataFrame({
                            'feature': feature_names,
                            'shap_value': shap_values[0],
                            'feature_value': customer_data.iloc[0].values
                        }).sort_values('shap_value', key=abs, ascending=False)
                        
                        print("Top SHAP contributions:")
                        for idx, row in contrib_df.head(5).iterrows():
                            direction = "increases" if row['shap_value'] > 0 else "decreases"
                            print(f"  {row['feature']}: {row['shap_value']:.4f} ({direction} churn risk)")
                        
                        explanations[model_name] = contrib_df.head(10)
                        
                    except Exception as e:
                        print(f"SHAP explanation failed: {e}")
                        print("Using fallback feature importance...")
                        # Fallback to feature importance
                        importance_df = get_tree_feature_importance(model, feature_names)
                        explanations[model_name] = importance_df.head(10)
                else:
                    # Fallback to feature importance
                    importance_df = get_tree_feature_importance(model, feature_names)
                    explanations[model_name] = importance_df.head(10)
            else:
                # Fallback to feature importance
                importance_df = get_tree_feature_importance(model, feature_names)
                explanations[model_name] = importance_df.head(10)
        
        print()
    
    return explanations

def create_interpretability_report(models, X_test, y_test, feature_names):
    """Create comprehensive interpretability report."""
    
    print("=== COMPREHENSIVE INTERPRETABILITY REPORT ===\n")
    
    report = {
        'global_importance': {},
        'shap_results': None,
        'sample_explanations': {}
    }
    
    # 1. Global feature importance
    print("1. Computing global feature importance...")
    report['global_importance'] = get_global_feature_importance(models, X_test, feature_names)
    
    # 2. Create global importance plot
    print("2. Creating global importance visualization...")
    create_global_importance_plot(report['global_importance'])
    
    # 3. SHAP global analysis (if available)
    if SHAP_AVAILABLE:
        print("3. Computing SHAP global importance...")
        report['shap_results'] = get_shap_global_importance(models, X_test, feature_names)
    else:
        print("3. SHAP not available - skipping SHAP global analysis")
    
    # 4. Sample local explanations
    print("4. Creating sample local explanations...")
    
    # Explain a few sample customers
    sample_indices = [0, 50, 100]  # Different customers
    for i, idx in enumerate(sample_indices):
        customer_data = X_test.iloc[idx:idx+1]
        customer_id = f"SAMPLE_{i+1}"
        
        explanation = explain_single_prediction(models, customer_data, feature_names, customer_id)
        report['sample_explanations'][customer_id] = explanation
    
    print("✅ Interpretability analysis complete!")
    
    return report

def save_interpretability_results(report):
    """Save interpretability results to files."""
    
    # Save global importance
    for model_name, importance_df in report['global_importance'].items():
        filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.csv"
        importance_df.to_csv(filename, index=False)
        print(f"Saved {filename}")
    
    # Save SHAP results if available
    if report['shap_results']:
        for model_name, shap_df in report['shap_results'].items():
            filename = f"shap_importance_{model_name.lower().replace(' ', '_')}.csv"
            shap_df.to_csv(filename, index=False)
            print(f"Saved {filename}")
    
    print("Interpretability results saved!")

def main():
    """Main interpretability analysis pipeline."""
    
    try:
        # Load models and data
        print("Loading models and test data...")
        models, X_test, y_test, feature_names = load_models_and_data()
        
        # Create comprehensive interpretability report
        report = create_interpretability_report(models, X_test, y_test, feature_names)
        
        # Save results
        save_interpretability_results(report)
        
        print(f"\n🎯 INTERPRETABILITY SUMMARY:")
        print(f"📊 Global importance plots: 'global_feature_importance.png'")
        if SHAP_AVAILABLE:
            print(f"🔍 SHAP plots: 'shap_importance_*.png'")
        else:
            print(f"⚠️  SHAP not available - using feature importance fallback")
        print(f"📋 CSV exports: 'feature_importance_*.csv'")
        
        return report
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run data preparation and model training first.")
        return None

if __name__ == "__main__":
    report = main()
"""
Customer Churn Prediction - Model Training
==========================================
Trains and evaluates Logistic Regression, Random Forest, and XGBoost models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load processed train/val/test data."""
    
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    # Separate features and targets
    feature_cols = [col for col in train_df.columns if col != 'Churn']
    
    X_train = train_df[feature_cols]
    y_train = train_df['Churn']
    
    X_val = val_df[feature_cols]
    y_val = val_df['Churn']
    
    X_test = test_df[feature_cols]
    y_test = test_df['Churn']
    
    print(f"Feature columns: {feature_cols}")
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    return (X_train, X_val, X_test), (y_train, y_val, y_test), feature_cols

def create_models():
    """Create model pipelines with preprocessing."""
    
    models = {}
    
    # 1. Logistic Regression (with scaling)
    models['Logistic Regression'] = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # 2. Random Forest (no scaling needed)
    models['Random Forest'] = Pipeline([
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    # 3. XGBoost (no scaling needed)
    models['XGBoost'] = Pipeline([
        ('classifier', XGBClassifier(
            random_state=42, 
            eval_metric='logloss',
            verbosity=0
        ))
    ])
    
    return models

def get_hyperparameter_grids():
    """Define hyperparameter grids for tuning."""
    
    param_grids = {
        'Logistic Regression': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__class_weight': [None, 'balanced']
        },
        
        'Random Forest': {
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__class_weight': [None, 'balanced']
        },
        
        'XGBoost': {
            'classifier__max_depth': [3, 6, 10],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__scale_pos_weight': [1, 2, 3]  # Handle class imbalance
        }
    }
    
    return param_grids

def tune_model(model, param_grid, X_train, y_train, X_val, y_val, model_name):
    """Tune hyperparameters using validation set."""
    
    print(f"Tuning {model_name}...")
    
    # Use GridSearch with validation set as test set
    # Combine train and val for cross-validation
    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])
    
    # Create validation split indices
    train_indices = list(range(len(X_train)))
    val_indices = list(range(len(X_train), len(X_combined)))
    
    cv_split = [(train_indices, val_indices)]
    
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=cv_split,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_combined, y_combined)
    
    print(f"Best parameters for {model_name}:")
    print(grid_search.best_params_)
    print(f"Best validation AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model on test set."""
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    metrics = {
        'Model': model_name,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC': auc
    }
    
    print(f"\n{model_name} Performance:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    return metrics, y_pred, y_pred_proba

def test_high_risk_customer(models, feature_cols, encoders):
    """Test models on a high-risk customer profile."""
    
    print("\n=== High-Risk Customer Test ===")
    
    # Create high-risk customer profile
    # Senior citizen + month-to-month + fiber optic + no support services + electronic check + $100+ monthly
    high_risk_customer = pd.DataFrame({
        'customerID': ['TEST_001'],
        'gender': ['Male'],
        'SeniorCitizen': ['1'],
        'Partner': ['No'],
        'Dependents': ['No'],
        'tenure': [3],
        'PhoneService': ['Yes'],
        'MultipleLines': ['No'],
        'InternetService': ['Fiber optic'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['No'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['No'],
        'StreamingMovies': ['No'],
        'Contract': ['Month-to-month'],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'],
        'MonthlyCharges': [105.0],
        'TotalCharges': [315.0]
    })
    
    print("High-risk customer profile:")
    print("- Senior citizen: Yes")
    print("- Contract: Month-to-month")
    print("- Internet: Fiber optic")
    print("- Support services: None")
    print("- Payment: Electronic check")
    print("- Monthly charges: $105")
    
    # Process the customer data (simplified feature engineering)
    test_customer = high_risk_customer.copy()
    
    # Add basic engineered features
    test_customer['tenure_bucket'] = '0-6m'
    test_customer['services_count'] = 2  # Phone + Internet
    test_customer['monthly_to_total_ratio'] = 105.0 / max(1, 3 * 105.0)
    test_customer['has_internet_no_support'] = 1
    test_customer['fiber_no_security'] = 1
    test_customer['expected_tenure'] = 24
    test_customer['clv'] = 105.0 * 24
    test_customer['clv_quartile'] = 'High'  # Simplified
    
    # Encode categorical variables (simplified - you'd use the actual encoders)
    categorical_mappings = {
        'gender': {'Female': 0, 'Male': 1},
        'Partner': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1},
        'PhoneService': {'No': 0, 'Yes': 1},
        'MultipleLines': {'No': 0, 'No phone service': 1, 'Yes': 2},
        'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
        'OnlineSecurity': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'OnlineBackup': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'DeviceProtection': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'StreamingTV': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'StreamingMovies': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
        'PaperlessBilling': {'No': 0, 'Yes': 1},
        'PaymentMethod': {'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1, 
                         'Electronic check': 2, 'Mailed check': 3},
        'tenure_bucket': {'0-6m': 0, '6-12m': 1, '12-24m': 2, '24m+': 3},
        'clv_quartile': {'Low': 0, 'Medium': 1, 'High': 2, 'Premium': 3}
    }
    
    for col, mapping in categorical_mappings.items():
        if col in test_customer.columns:
            test_customer[col] = test_customer[col].map(mapping)
    
    # Select features that match training data
    test_features = test_customer[feature_cols]
    
    print(f"\nPredictions for high-risk customer:")
    for name, model in models.items():
        try:
            prob = model.predict_proba(test_features)[0, 1]
            print(f"{name}: {prob:.1%} churn probability")
            
            if prob > 0.6:
                print(f"  ✅ PASS - {name} correctly identifies high risk")
            else:
                print(f"  ❌ FAIL - {name} shows {prob:.1%} (expected >60%)")
        except Exception as e:
            print(f"{name}: Error - {e}")

def save_models(models, feature_cols):
    """Save trained models and metadata."""
    
    # Save individual models
    for name, model in models.items():
        filename = name.lower().replace(' ', '_') + '.pkl'
        joblib.dump(model, f'models/{filename}')
        print(f"Saved {name} to models/{filename}")
    
    # Save feature names
    joblib.dump(feature_cols, 'models/feature_columns.pkl')
    print("Saved feature columns to models/feature_columns.pkl")

def create_model_comparison_plot(results_df):
    """Create model comparison visualization."""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Metrics comparison
    metrics = ['Precision', 'Recall', 'F1', 'AUC']
    x = np.arange(len(results_df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        axes[0].bar(x + i*width, results_df[metric], width, 
                   label=metric, alpha=0.8)
    
    axes[0].set_xlabel('Models')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Model Performance Comparison')
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(results_df['Model'], rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # Plot 2: AUC focus
    bars = axes[1].bar(results_df['Model'], results_df['AUC'], 
                      color=['skyblue', 'lightcoral', 'lightgreen'], 
                      alpha=0.8, edgecolor='black')
    axes[1].set_title('AUC-ROC Comparison')
    axes[1].set_ylabel('AUC Score')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.5, 1.0)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """Main model training pipeline."""
    
    print("=== Customer Churn Model Training ===\n")
    
    try:
        # Load data
        print("1. Loading processed data...")
        (X_train, X_val, X_test), (y_train, y_val, y_test), feature_cols = load_data()
        
        # Load encoders for testing
        encoders = joblib.load('models/encoders.pkl')
        
        # Create models
        print("\n2. Creating model pipelines...")
        models = create_models()
        param_grids = get_hyperparameter_grids()
        
        # Train and tune models
        print("\n3. Training and tuning models...")
        trained_models = {}
        results = []
        
        for name, model in models.items():
            print(f"\n--- {name} ---")
            
            # Tune hyperparameters
            best_model = tune_model(
                model, param_grids[name], 
                X_train, y_train, X_val, y_val, name
            )
            
            # Evaluate on test set
            metrics, y_pred, y_pred_proba = evaluate_model(
                best_model, X_test, y_test, name
            )
            
            trained_models[name] = best_model
            results.append(metrics)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        print(f"\n=== FINAL MODEL COMPARISON ===")
        print(results_df.round(4))
        
        # Test high-risk customer
        test_high_risk_customer(trained_models, feature_cols, encoders)
        
        # Create comparison plot
        print(f"\n4. Creating model comparison plot...")
        fig = create_model_comparison_plot(results_df)
        
        # Save models
        print(f"\n5. Saving trained models...")
        save_models(trained_models, feature_cols)
        
        # Save results
        results_df.to_csv('model_results.csv', index=False)
        
        print(f"\n✅ Model training complete!")
        print(f"📊 Comparison plot saved as 'model_comparison.png'")
        print(f"📋 Results saved as 'model_results.csv'")
        
        # Performance check
        best_model_idx = results_df['AUC'].idxmax()
        best_model_name = results_df.loc[best_model_idx, 'Model']
        best_auc = results_df.loc[best_model_idx, 'AUC']
        best_recall = results_df.loc[best_model_idx, 'Recall']
        
        print(f"\n🏆 Best model: {best_model_name}")
        print(f"   AUC: {best_auc:.4f}")
        print(f"   Recall: {best_recall:.4f}")
        
        if best_auc >= 0.80 and best_recall >= 0.60:
            print("   ✅ Performance targets met!")
        else:
            print("   ⚠️  Performance below targets - consider:")
            print("     - More feature engineering")
            print("     - Handling class imbalance")
            print("     - Advanced hyperparameter tuning")
        
        return trained_models, results_df
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run data preparation first.")
        return None, None

if __name__ == "__main__":
    models, results = main()
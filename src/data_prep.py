"""
Customer Churn Prediction - Data Preparation
============================================
This script handles data loading, cleaning, feature engineering, and CLV calculation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def load_and_clean_data():
    """Load and clean the IBM Telco Customer Churn dataset."""
    
    # Load data
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Handle TotalCharges - convert to numeric and handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Strategy: Fill missing TotalCharges with tenure * MonthlyCharges
    # This assumes customers with missing TotalCharges are very new
    missing_mask = df['TotalCharges'].isna()
    df.loc[missing_mask, 'TotalCharges'] = (
        df.loc[missing_mask, 'tenure'] * df.loc[missing_mask, 'MonthlyCharges']
    )
    
    print(f"Filled {missing_mask.sum()} missing TotalCharges values")
    
    # Convert SeniorCitizen to string for consistency
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
    
    return df

def engineer_features(df):
    """Create business-driven features."""
    
    df = df.copy()
    
    # 1. Tenure buckets
    df['tenure_bucket'] = pd.cut(
        df['tenure'],
        bins=[0, 6, 12, 24, float('inf')],
        labels=['0-6m', '6-12m', '12-24m', '24m+'],
        right=False
    ).astype(str)
    
    # 2. Services count
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    # Count active services (not "No" or "No internet service")
    df['services_count'] = 0
    for col in service_cols:
        if col in df.columns:
            df['services_count'] += (
                (df[col] == 'Yes') | 
                (col == 'InternetService' and df[col].isin(['DSL', 'Fiber optic']))
            ).astype(int)
    
    # 3. Monthly to total ratio
    df['monthly_to_total_ratio'] = df['MonthlyCharges'] / np.maximum(
        1, df['tenure'] * df['MonthlyCharges']
    )
    
    # 4. Service flags
    df['has_internet_no_support'] = (
        (df['InternetService'].isin(['DSL', 'Fiber optic'])) &
        (df['TechSupport'] == 'No')
    ).astype(int)
    
    df['fiber_no_security'] = (
        (df['InternetService'] == 'Fiber optic') &
        (df['OnlineSecurity'] == 'No')
    ).astype(int)
    
    return df

def calculate_clv(df, expected_tenure_months=24):
    """
    Calculate Customer Lifetime Value.
    
    Assumption: Expected tenure is 24 months for all customers.
    This is a simplified approach - in practice, you might use 
    survival analysis or segment-specific estimates.
    """
    
    df = df.copy()
    df['expected_tenure'] = expected_tenure_months
    df['clv'] = df['MonthlyCharges'] * df['expected_tenure']
    
    # Create CLV quartiles
    df['clv_quartile'] = pd.qcut(
        df['clv'],
        q=4,
        labels=['Low', 'Medium', 'High', 'Premium']
    ).astype(str)
    
    print(f"CLV Statistics:")
    print(df['clv'].describe())
    print(f"\nCLV Quartiles:")
    print(df['clv_quartile'].value_counts().sort_index())
    
    return df

def encode_categorical_features(df):
    """Encode categorical variables using LabelEncoder."""
    
    df = df.copy()
    
    # Identify categorical columns (excluding target)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'Churn' in categorical_cols:
        categorical_cols.remove('Churn')
    
    # Initialize encoders dictionary
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
        # Print encoding mapping for verification
        print(f"{col} encoding:")
        for i, class_name in enumerate(le.classes_):
            print(f"  {class_name}: {i}")
        print()
    
    # Encode target variable
    target_encoder = LabelEncoder()
    df['Churn'] = target_encoder.fit_transform(df['Churn'])
    encoders['Churn'] = target_encoder
    
    return df, encoders

def split_data(df, test_size=0.2, val_size=0.2, random_state=42):
    """Split data into train/val/test with stratification."""
    
    # Separate features and target
    X = df.drop(['Churn'], axis=1)
    y = df['Churn']
    
    # First split: train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train and val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    print(f"Data split sizes:")
    print(f"Train: {len(X_train)} ({len(X_train)/len(df):.1%})")
    print(f"Val: {len(X_val)} ({len(X_val)/len(df):.1%})")
    print(f"Test: {len(X_test)} ({len(X_test)/len(df):.1%})")
    
    print(f"\nChurn distribution:")
    print(f"Train: {y_train.mean():.3f}")
    print(f"Val: {y_val.mean():.3f}")
    print(f"Test: {y_test.mean():.3f}")
    
    return (X_train, X_val, X_test), (y_train, y_val, y_test)

def save_processed_data(X_splits, y_splits, encoders):
    """Save processed data and encoders."""
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    X_train, X_val, X_test = X_splits
    y_train, y_val, y_test = y_splits
    
    # Combine features and targets
    train_df = X_train.copy()
    train_df['Churn'] = y_train
    
    val_df = X_val.copy()
    val_df['Churn'] = y_val
    
    test_df = X_test.copy()
    test_df['Churn'] = y_test
    
    # Save splits
    train_df.to_csv('data/processed/train.csv', index=False)
    val_df.to_csv('data/processed/val.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    # Save encoders
    import joblib
    joblib.dump(encoders, 'models/encoders.pkl')
    
    print("Processed data saved to data/processed/")
    print("Encoders saved to models/encoders.pkl")

def main():
    """Main data preparation pipeline."""
    
    print("=== Customer Churn Data Preparation ===\n")
    
    # Step 1: Load and clean data
    print("1. Loading and cleaning data...")
    df = load_and_clean_data()
    
    # Step 2: Engineer features
    print("\n2. Engineering features...")
    df = engineer_features(df)
    
    # Step 3: Calculate CLV
    print("\n3. Calculating CLV...")
    df = calculate_clv(df, expected_tenure_months=24)
    
    # Step 4: Encode categorical features
    print("\n4. Encoding categorical features...")
    df, encoders = encode_categorical_features(df)
    
    # Step 5: Split data
    print("\n5. Splitting data...")
    X_splits, y_splits = split_data(df)
    
    # Step 6: Save processed data
    print("\n6. Saving processed data...")
    save_processed_data(X_splits, y_splits, encoders)
    
    print("\n✅ Data preparation complete!")
    
    # Show final dataset info
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Features: {[col for col in df.columns if col != 'Churn']}")

if __name__ == "__main__":
    main()
"""
Prediction Utilities
====================
Utilities for making predictions on new customer data.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictor:
    """Class for making churn predictions on new customer data."""
    
    def __init__(self, models_path='models/'):
        """
        Initialize the predictor with trained models.
        
        Args:
            models_path (str): Path to the directory containing trained models
        """
        self.models_path = models_path
        self.models = {}
        self.encoders = None
        self.feature_columns = None
        self._load_models()
    
    def _load_models(self):
        """Load trained models and preprocessing objects."""
        
        try:
            # Load models
            self.models = {
                'Logistic Regression': joblib.load(f'{self.models_path}logistic_regression.pkl'),
                'Random Forest': joblib.load(f'{self.models_path}random_forest.pkl'),
                'XGBoost': joblib.load(f'{self.models_path}xgboost.pkl')
            }
            
            # Load encoders and feature columns
            self.encoders = joblib.load(f'{self.models_path}encoders.pkl')
            self.feature_columns = joblib.load(f'{self.models_path}feature_columns.pkl')
            
            print("✅ Models loaded successfully!")
            print(f"Available models: {list(self.models.keys())}")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading models: {e}")
            print("Please ensure models are trained and saved in the models/ directory")
            raise
    
    def preprocess_customer_data(self, customer_data):
        """
        Preprocess customer data to match training format.
        
        Args:
            customer_data (dict or pd.DataFrame): Raw customer data
            
        Returns:
            pd.DataFrame: Processed features ready for prediction
        """
        
        # Convert to DataFrame if needed
        if isinstance(customer_data, dict):
            df = pd.DataFrame([customer_data])
        else:
            df = customer_data.copy()
        
        # Apply feature engineering
        df = self._engineer_features(df)
        
        # Encode categorical variables
        df = self._encode_features(df)
        
        # Select only features used in training
        df = df[self.feature_columns]
        
        return df
    
    def _engineer_features(self, df):
        """Apply the same feature engineering as training."""
        
        df = df.copy()
        
        # Handle TotalCharges if missing
        if 'TotalCharges' not in df.columns or df['TotalCharges'].isna().any():
            if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
                df['TotalCharges'] = df['tenure'] * df['MonthlyCharges']
            else:
                df['TotalCharges'] = df.get('MonthlyCharges', 0) * df.get('tenure', 1)
        
        # Tenure buckets
        def get_tenure_bucket(tenure):
            if tenure <= 6:
                return '0-6m'
            elif tenure <= 12:
                return '6-12m'
            elif tenure <= 24:
                return '12-24m'
            else:
                return '24m+'
        
        df['tenure_bucket'] = df['tenure'].apply(get_tenure_bucket)
        
        # Services count
        service_cols = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        services_count = 0
        for col in service_cols:
            if col in df.columns:
                if col == 'InternetService':
                    services_count += (df[col].isin(['DSL', 'Fiber optic'])).astype(int)
                else:
                    services_count += (df[col] == 'Yes').astype(int)
        
        df['services_count'] = services_count
        
        # Monthly to total ratio
        df['monthly_to_total_ratio'] = df['MonthlyCharges'] / np.maximum(
            1, df['tenure'] * df['MonthlyCharges']
        )
        
        # Service flags
        df['has_internet_no_support'] = (
            (df['InternetService'].isin(['DSL', 'Fiber optic'])) &
            (df.get('TechSupport', 'No') == 'No')
        ).astype(int)
        
        df['fiber_no_security'] = (
            (df['InternetService'] == 'Fiber optic') &
            (df.get('OnlineSecurity', 'No') == 'No')
        ).astype(int)
        
        # CLV calculation (expected tenure assumption)
        expected_tenure = 24
        df['expected_tenure'] = expected_tenure
        df['clv'] = df['MonthlyCharges'] * expected_tenure
        
        # CLV quartile (simplified)
        def get_clv_quartile(clv):
            if clv < 1000:
                return 'Low'
            elif clv < 2000:
                return 'Medium'
            elif clv < 3000:
                return 'High'
            else:
                return 'Premium'
        
        df['clv_quartile'] = df['clv'].apply(get_clv_quartile)
        
        return df
    
    def _encode_features(self, df):
        """Encode categorical features using training encoders."""
        
        df_encoded = df.copy()
        
        # Categorical encoding mappings (must match training exactly)
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
            'PaymentMethod': {
                'Bank transfer (automatic)': 0, 
                'Credit card (automatic)': 1,
                'Electronic check': 2, 
                'Mailed check': 3
            },
            'tenure_bucket': {'0-6m': 0, '6-12m': 1, '12-24m': 2, '24m+': 3},
            'clv_quartile': {'Low': 0, 'Medium': 1, 'High': 2, 'Premium': 3}
        }
        
        # Apply encoding
        for col, mapping in categorical_mappings.items():
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].map(mapping)
                # Fill any missing mappings with 0
                df_encoded[col] = df_encoded[col].fillna(0)
        
        # Convert SeniorCitizen to numeric if it's not already
        if 'SeniorCitizen' in df_encoded.columns:
            if df_encoded['SeniorCitizen'].dtype == 'object':
                df_encoded['SeniorCitizen'] = df_encoded['SeniorCitizen'].map({'0': 0, '1': 1, 'No': 0, 'Yes': 1})
            df_encoded['SeniorCitizen'] = df_encoded['SeniorCitizen'].astype(int)
        
        return df_encoded
    
    def predict_churn_probability(self, customer_data, model_name='ensemble'):
        """
        Predict churn probability for a customer.
        
        Args:
            customer_data (dict or pd.DataFrame): Customer data
            model_name (str): Model to use ('Logistic Regression', 'Random Forest', 
                            'XGBoost', or 'ensemble')
                            
        Returns:
            float: Churn probability (0-1)
        """
        
        # Preprocess data
        processed_data = self.preprocess_customer_data(customer_data)
        
        if model_name == 'ensemble':
            # Use ensemble of all models
            probabilities = []
            for name, model in self.models.items():
                prob = model.predict_proba(processed_data)[0, 1]
                probabilities.append(prob)
            return np.mean(probabilities)
        
        elif model_name in self.models:
            # Use specific model
            return self.models[model_name].predict_proba(processed_data)[0, 1]
        
        else:
            raise ValueError(f"Model '{model_name}' not available. "
                           f"Available models: {list(self.models.keys()) + ['ensemble']}")
    
    def predict_all_models(self, customer_data):
        """
        Get predictions from all models.
        
        Args:
            customer_data (dict or pd.DataFrame): Customer data
            
        Returns:
            dict: Predictions from all models + ensemble
        """
        
        processed_data = self.preprocess_customer_data(customer_data)
        
        results = {}
        probabilities = []
        
        for name, model in self.models.items():
            prob = model.predict_proba(processed_data)[0, 1]
            results[name] = prob
            probabilities.append(prob)
        
        # Add ensemble prediction
        results['Ensemble'] = np.mean(probabilities)
        
        return results
    
    def get_risk_level(self, probability):
        """Convert probability to risk level."""
        if probability < 0.3:
            return "Low Risk", "🟢"
        elif probability < 0.6:
            return "Medium Risk", "🟡"
        else:
            return "High Risk", "🔴"
    
    def calculate_clv(self, customer_data, expected_tenure=24):
        """Calculate Customer Lifetime Value."""
        
        if isinstance(customer_data, dict):
            monthly_charges = customer_data.get('MonthlyCharges', 0)
        else:
            monthly_charges = customer_data.get('MonthlyCharges', pd.Series([0])).iloc[0]
        
        return monthly_charges * expected_tenure
    
    def get_prediction_summary(self, customer_data):
        """
        Get comprehensive prediction summary for a customer.
        
        Args:
            customer_data (dict or pd.DataFrame): Customer data
            
        Returns:
            dict: Complete prediction summary
        """
        
        # Get predictions from all models
        predictions = self.predict_all_models(customer_data)
        
        # Calculate CLV
        clv = self.calculate_clv(customer_data)
        
        # Get risk levels
        risk_summary = {}
        for model_name, prob in predictions.items():
            risk_level, icon = self.get_risk_level(prob)
            risk_summary[model_name] = {
                'probability': prob,
                'risk_level': risk_level,
                'icon': icon
            }
        
        # Customer info
        if isinstance(customer_data, dict):
            customer_info = customer_data
        else:
            customer_info = customer_data.iloc[0].to_dict()
        
        summary = {
            'customer_info': customer_info,
            'predictions': risk_summary,
            'clv': clv,
            'recommendations': self._get_recommendations(predictions['Ensemble'], clv)
        }
        
        return summary
    
    def _get_recommendations(self, churn_prob, clv):
        """Generate retention recommendations based on risk and value."""
        
        recommendations = []
        
        # High-risk recommendations
        if churn_prob > 0.6:
            recommendations.extend([
                "🚨 HIGH PRIORITY: Immediate personal outreach within 24 hours",
                "💰 Offer contract upgrade incentives or loyalty discounts",
                "🛠️ Provide premium tech support and service optimization",
                "📞 Schedule executive-level retention call"
            ])
        elif churn_prob > 0.3:
            recommendations.extend([
                "⚠️ MONITOR CLOSELY: Schedule quarterly check-in calls",
                "📊 Provide usage insights and service optimization tips",
                "🎯 Offer targeted value-added services",
                "📈 Include in proactive retention campaigns"
            ])
        else:
            recommendations.extend([
                "✅ LOW RISK: Continue standard customer care",
                "📋 Include in regular satisfaction surveys",
                "🚀 Present upgrade and expansion opportunities",
                "📱 Monitor usage patterns for early warning signs"
            ])
        
        # CLV-based recommendations
        if clv > 3000:
            recommendations.append("💎 PREMIUM CUSTOMER: Assign dedicated account manager")
        elif clv > 2000:
            recommendations.append("⭐ HIGH VALUE: Priority customer service queue")
        
        return recommendations

def create_sample_customer():
    """Create a sample high-risk customer for testing."""
    
    return {
        'customerID': 'SAMPLE_001',
        'gender': 'Male',
        'SeniorCitizen': '1',
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 3,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 105.0,
        'TotalCharges': 315.0
    }

def test_predictor():
    """Test the churn predictor with sample data."""
    
    print("=== TESTING CHURN PREDICTOR ===\n")
    
    try:
        # Initialize predictor
        predictor = ChurnPredictor()
        
        # Create sample customer (high-risk profile)
        sample_customer = create_sample_customer()
        
        print("Sample Customer Profile:")
        print(f"- Senior Citizen: {'Yes' if sample_customer['SeniorCitizen'] == '1' else 'No'}")
        print(f"- Contract: {sample_customer['Contract']}")
        print(f"- Internet Service: {sample_customer['InternetService']}")
        print(f"- Tech Support: {sample_customer['TechSupport']}")
        print(f"- Payment Method: {sample_customer['PaymentMethod']}")
        print(f"- Monthly Charges: ${sample_customer['MonthlyCharges']}")
        print(f"- Tenure: {sample_customer['tenure']} months")
        
        print("\n--- Individual Model Predictions ---")
        
        # Test individual models
        for model_name in ['Logistic Regression', 'Random Forest', 'XGBoost']:
            prob = predictor.predict_churn_probability(sample_customer, model_name)
            risk_level, icon = predictor.get_risk_level(prob)
            print(f"{model_name}: {prob:.1%} - {icon} {risk_level}")
        
        # Test ensemble
        ensemble_prob = predictor.predict_churn_probability(sample_customer, 'ensemble')
        risk_level, icon = predictor.get_risk_level(ensemble_prob)
        print(f"Ensemble: {ensemble_prob:.1%} - {icon} {risk_level}")
        
        print("\n--- Complete Prediction Summary ---")
        
        # Get comprehensive summary
        summary = predictor.get_prediction_summary(sample_customer)
        
        print(f"Customer CLV: ${summary['clv']:,.0f}")
        print(f"Best Model Prediction: {max(summary['predictions'].values(), key=lambda x: x['probability'])}")
        
        print("\nRecommendations:")
        for rec in summary['recommendations']:
            print(f"  {rec}")
        
        # Validation check
        print(f"\n--- Validation Check ---")
        if ensemble_prob > 0.6:
            print("✅ PASS: High-risk customer correctly identified (>60% churn probability)")
        else:
            print(f"❌ FAIL: Expected >60% churn probability, got {ensemble_prob:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing predictor: {e}")
        return False

def batch_predict(customer_data_file, output_file='batch_predictions.csv'):
    """
    Make predictions on a batch of customers.
    
    Args:
        customer_data_file (str): Path to CSV file with customer data
        output_file (str): Path for output predictions file
    """
    
    print(f"=== BATCH PREDICTION ===\n")
    
    try:
        # Initialize predictor
        predictor = ChurnPredictor()
        
        # Load customer data
        customers_df = pd.read_csv(customer_data_file)
        print(f"Loaded {len(customers_df)} customers from {customer_data_file}")
        
        # Make predictions
        predictions = []
        
        for idx, row in customers_df.iterrows():
            customer_data = row.to_dict()
            
            try:
                # Get predictions from all models
                model_predictions = predictor.predict_all_models(customer_data)
                
                # Calculate CLV
                clv = predictor.calculate_clv(customer_data)
                
                # Get risk level
                ensemble_prob = model_predictions['Ensemble']
                risk_level, _ = predictor.get_risk_level(ensemble_prob)
                
                # Store results
                result = {
                    'customerID': customer_data.get('customerID', f'CUSTOMER_{idx}'),
                    'churn_probability': ensemble_prob,
                    'risk_level': risk_level,
                    'clv': clv,
                    'logistic_prob': model_predictions['Logistic Regression'],
                    'rf_prob': model_predictions['Random Forest'],
                    'xgb_prob': model_predictions['XGBoost']
                }
                
                predictions.append(result)
                
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1} customers...")
                    
            except Exception as e:
                print(f"Error processing customer {idx}: {e}")
                continue
        
        # Save results
        results_df = pd.DataFrame(predictions)
        results_df = results_df.sort_values('churn_probability', ascending=False)
        results_df.to_csv(output_file, index=False)
        
        print(f"\n✅ Batch prediction complete!")
        print(f"📄 Results saved to {output_file}")
        print(f"📊 Processed {len(results_df)} customers successfully")
        
        # Summary statistics
        high_risk = (results_df['churn_probability'] > 0.6).sum()
        medium_risk = ((results_df['churn_probability'] > 0.3) & 
                      (results_df['churn_probability'] <= 0.6)).sum()
        low_risk = (results_df['churn_probability'] <= 0.3).sum()
        
        print(f"\n📈 Risk Distribution:")
        print(f"  🔴 High Risk (>60%): {high_risk} customers")
        print(f"  🟡 Medium Risk (30-60%): {medium_risk} customers")
        print(f"  🟢 Low Risk (<30%): {low_risk} customers")
        
        return results_df
        
    except Exception as e:
        print(f"❌ Error in batch prediction: {e}")
        return None

def main():
    """Main function for testing prediction utilities."""
    
    print("=== CHURN PREDICTION UTILITIES ===\n")
    
    # Test the predictor
    success = test_predictor()
    
    if success:
        print(f"\n🎯 Predictor is working correctly!")
        print(f"💡 You can now use it in your Streamlit app or for batch predictions.")
        
        # Example usage
        print(f"\n--- Example Usage ---")
        print("""
# Initialize predictor
predictor = ChurnPredictor()

# Single prediction
customer = {...}  # Customer data dictionary
probability = predictor.predict_churn_probability(customer)
summary = predictor.get_prediction_summary(customer)

# Batch prediction
results = batch_predict('customers.csv', 'predictions.csv')
        """)
    else:
        print(f"\n❌ Please fix the errors and run model training first.")

if __name__ == "__main__":
    main()
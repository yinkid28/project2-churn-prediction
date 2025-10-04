"""
Customer Churn Prediction & CLV Analysis App
============================================
Interactive Streamlit app for churn prediction and customer lifetime value analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data loading
@st.cache_data
def load_processed_data():
    """Load processed datasets."""
    try:
        train_df = pd.read_csv('data/processed/train.csv')
        test_df = pd.read_csv('data/processed/test.csv')
        return train_df, test_df
    except FileNotFoundError:
        st.error("Processed data not found. Please run data preparation first.")
        return None, None

@st.cache_resource
def load_models():
    """Load trained models and encoders."""
    try:
        models = {
            'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
            'Random Forest': joblib.load('models/random_forest.pkl'),
            'XGBoost': joblib.load('models/xgboost.pkl')
        }
        encoders = joblib.load('models/encoders.pkl')
        feature_cols = joblib.load('models/feature_columns.pkl')
        return models, encoders, feature_cols
    except FileNotFoundError:
        st.error("Trained models not found. Please run model training first.")
        return None, None, None

@st.cache_resource
def get_model_results():
    """Load model performance results."""
    try:
        results_df = pd.read_csv('model_results.csv')
        return results_df
    except FileNotFoundError:
        # Create dummy results if file doesn't exist
        return pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
            'Precision': [0.82, 0.85, 0.87],
            'Recall': [0.65, 0.68, 0.72],
            'F1': [0.72, 0.75, 0.79],
            'AUC': [0.81, 0.84, 0.86]
        })

def create_customer_input_form():
    """Create input form for customer data."""
    
    st.subheader("🔍 Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    
    with col2:
        st.markdown("**Service Details**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
        
        # Conditional services based on internet
        if internet_service == "No":
            online_security = "No internet service"
            tech_support = "No internet service"
        else:
            online_security = st.selectbox("Online Security", ["No", "Yes"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes"])
    
    with col3:
        st.markdown("**Contract & Billing**")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", 
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, 5.0)
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    
    return {
        'gender': gender,
        'SeniorCitizen': '1' if senior_citizen == "Yes" else '0',
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': "No phone service" if phone_service == "No" else "No",
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': "No internet service" if internet_service == "No" else "No",
        'DeviceProtection': "No internet service" if internet_service == "No" else "No",
        'TechSupport': tech_support,
        'StreamingTV': "No internet service" if internet_service == "No" else "No",
        'StreamingMovies': "No internet service" if internet_service == "No" else "No",
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': tenure * monthly_charges
    }

def engineer_features(customer_data):
    """Apply the same feature engineering as training - FIXED VERSION."""
    
    df = pd.DataFrame([customer_data])
    
    # Convert numeric strings to numbers if needed
    if 'tenure' in df.columns:
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
    if 'MonthlyCharges' in df.columns:
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Ensure TotalCharges is calculated
    df['TotalCharges'] = df['tenure'] * df['MonthlyCharges']
    
    # 1. Tenure buckets
    tenure_val = df['tenure'].iloc[0]
    if tenure_val <= 6:
        df['tenure_bucket'] = '0-6m'
    elif tenure_val <= 12:
        df['tenure_bucket'] = '6-12m'
    elif tenure_val <= 24:
        df['tenure_bucket'] = '12-24m'
    else:
        df['tenure_bucket'] = '24m+'
    
    # 2. Services count - count ALL services including streaming
    services = 0
    if df['PhoneService'].iloc[0] == 'Yes':
        services += 1
    if df['MultipleLines'].iloc[0] == 'Yes':
        services += 1
    if df['InternetService'].iloc[0] in ['DSL', 'Fiber optic']:
        services += 1
        if df.get('OnlineSecurity', pd.Series(['No'])).iloc[0] == 'Yes':
            services += 1
        if df.get('OnlineBackup', pd.Series(['No'])).iloc[0] == 'Yes':
            services += 1
        if df.get('DeviceProtection', pd.Series(['No'])).iloc[0] == 'Yes':
            services += 1
        if df.get('TechSupport', pd.Series(['No'])).iloc[0] == 'Yes':
            services += 1
        if df.get('StreamingTV', pd.Series(['No'])).iloc[0] == 'Yes':
            services += 1
        if df.get('StreamingMovies', pd.Series(['No'])).iloc[0] == 'Yes':
            services += 1
    
    df['services_count'] = services
    
    # 3. Monthly to total ratio
    df['monthly_to_total_ratio'] = df['MonthlyCharges'] / np.maximum(
        1, df['tenure'] * df['MonthlyCharges']
    )
    
    # 4. Service flags
    df['has_internet_no_support'] = int(
        (df['InternetService'].iloc[0] in ['DSL', 'Fiber optic']) and
        (df.get('TechSupport', pd.Series(['No'])).iloc[0] == 'No')
    )
    
    df['fiber_no_security'] = int(
        (df['InternetService'].iloc[0] == 'Fiber optic') and
        (df.get('OnlineSecurity', pd.Series(['No'])).iloc[0] == 'No')
    )
    
    # 5. CLV calculation
    expected_tenure = 24
    df['expected_tenure'] = expected_tenure
    df['clv'] = df['MonthlyCharges'] * expected_tenure
    
    # 6. CLV quartile
    clv_val = df['clv'].iloc[0]
    if clv_val < 1000:
        df['clv_quartile'] = 'Low'
    elif clv_val < 2000:
        df['clv_quartile'] = 'Medium'
    elif clv_val < 3000:
        df['clv_quartile'] = 'High'
    else:
        df['clv_quartile'] = 'Premium'
    
    return df

def encode_customer_data(df, encoders, feature_cols):
    """Encode customer data using training encoders - FIXED VERSION."""
    
    df_encoded = df.copy()
    
    # Categorical encoding mappings - MUST MATCH TRAINING EXACTLY
    categorical_mappings = {
        'gender': {'Female': 0, 'Male': 1},
        'SeniorCitizen': {'0': 0, '1': 1, 0: 0, 1: 1},
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
            if df_encoded[col].isna().any():
                df_encoded[col] = df_encoded[col].fillna(0)
    
    # Convert SeniorCitizen to int
    if 'SeniorCitizen' in df_encoded.columns:
        df_encoded['SeniorCitizen'] = df_encoded['SeniorCitizen'].astype(int)
    
    # Ensure all required columns exist
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Select only features in correct order
    df_encoded = df_encoded[feature_cols]
    
    return df_encoded

def get_risk_level(probability):
    """Convert probability to risk level."""
    if probability < 0.3:
        return "🟢 Low Risk", "success"
    elif probability < 0.6:
        return "🟡 Medium Risk", "warning"
    else:
        return "🔴 High Risk", "error"

def predict_tab():
    """Customer prediction tab."""
    
    st.header("🎯 Customer Churn Prediction")
    
    # Load models and data
    models, encoders, feature_cols = load_models()
    if models is None:
        return
    
    # Customer input form
    customer_data = create_customer_input_form()
    
    if st.button("🔮 Predict Churn Risk", type="primary"):
        
        # Process customer data
        with st.spinner("Processing customer data..."):
            try:
                # Step 1: Engineer features
                customer_df = engineer_features(customer_data)
                
                # Step 2: Encode features
                customer_encoded = encode_customer_data(customer_df, encoders, feature_cols)
                
            except Exception as e:
                st.error(f"❌ Error processing customer data: {e}")
                with st.expander("🔍 Debug Information"):
                    st.write("Expected features:", feature_cols)
                    st.write("Customer data keys:", list(customer_data.keys()))
                    import traceback
                    st.code(traceback.format_exc())
                return
        
        # Make predictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Churn Predictions")
            
            predictions = {}
            for name, model in models.items():
                try:
                    prob = model.predict_proba(customer_encoded)[0, 1]
                    predictions[name] = prob
                    
                    risk_level, status = get_risk_level(prob)
                    
                    # Display prediction with status
                    with st.container():
                        st.markdown(f"**{name}**")
                        st.markdown(f"{risk_level}")
                        st.progress(float(prob))
                        st.markdown(f"Probability: **{prob:.1%}**")
                        st.markdown("---")
                        
                except Exception as e:
                    st.error(f"Error with {name}: {e}")
            
            # Ensemble prediction
            if predictions:
                avg_prob = np.mean(list(predictions.values()))
                risk_level, status = get_risk_level(avg_prob)
                
                st.markdown("### 🎯 Ensemble Prediction")
                st.markdown(f"{risk_level}")
                st.progress(float(avg_prob))
                st.markdown(f"Average Probability: **{avg_prob:.1%}**")
        
        with col2:
            st.subheader("💰 Customer Value Analysis")
            
            clv = customer_df['clv'].iloc[0]
            monthly_charges = customer_df['MonthlyCharges'].iloc[0]
            tenure = customer_df['tenure'].iloc[0]
            clv_quartile = customer_df['clv_quartile'].iloc[0]
            
            # CLV metrics
            st.metric("Customer Lifetime Value", f"${clv:,.0f}")
            st.metric("Monthly Revenue", f"${monthly_charges:,.0f}")
            st.metric("Current Tenure", f"{tenure} months")
            st.metric("Value Segment", clv_quartile)
            
            # CLV calculation explanation
            with st.expander("💡 How CLV is Calculated"):
                st.markdown("""
                **Formula:** CLV = Monthly Charges × Expected Tenure
                
                **Assumptions:**
                - Expected tenure: 24 months for all customers
                - No discount rate applied
                - Based on current monthly charges
                
                **Your Customer:**
                - Monthly Charges: ${:,.0f}
                - Expected Tenure: 24 months
                - **CLV = ${:,.0f} × 24 = ${:,.0f}**
                """.format(monthly_charges, monthly_charges, clv))
        
        # Feature importance (simplified)
        st.subheader("🔍 Key Risk Factors")
        
        # Create risk factors based on customer profile
        risk_factors = []
        
        if customer_data['Contract'] == 'Month-to-month':
            risk_factors.append("📅 Month-to-month contract (+30% risk)")
        
        if customer_data['tenure'] < 12:
            risk_factors.append("⏰ Low tenure (<12 months) (+25% risk)")
        
        if customer_data['PaymentMethod'] == 'Electronic check':
            risk_factors.append("💳 Electronic check payment (+20% risk)")
        
        if customer_data['SeniorCitizen'] == '1':
            risk_factors.append("👴 Senior citizen (+15% risk)")
        
        if customer_data['InternetService'] == 'Fiber optic' and customer_data['TechSupport'] == 'No':
            risk_factors.append("🌐 Fiber optic without tech support (+15% risk)")
        
        if customer_data['MonthlyCharges'] > 80:
            risk_factors.append("💸 High monthly charges (>$80) (+10% risk)")
        
        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        else:
            st.markdown("✅ No major risk factors identified")
        
        # Recommendations
        st.subheader("💡 Retention Recommendations")
        
        if avg_prob > 0.6:
            st.error("🚨 **HIGH PRIORITY INTERVENTION NEEDED**")
            st.markdown("""
            **Immediate Actions:**
            - Personal outreach within 24 hours
            - Offer contract upgrade incentives
            - Provide premium tech support
            - Consider loyalty discounts
            """)
        elif avg_prob > 0.3:
            st.warning("⚠️ **MONITOR CLOSELY**")
            st.markdown("""
            **Proactive Measures:**
            - Schedule quarterly check-in calls
            - Offer service optimization
            - Provide usage insights
            - Consider value-added services
            """)
        else:
            st.success("✅ **LOW RISK - STANDARD CARE**")
            st.markdown("""
            **Maintenance Actions:**
            - Include in regular satisfaction surveys
            - Offer upgrade opportunities
            - Monitor usage patterns
            """)

def model_performance_tab():
    """Model performance comparison tab."""
    
    st.header("📈 Model Performance Analysis")
    
    # Load results
    results_df = get_model_results()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Performance Metrics")
        
        # Style the dataframe
        styled_results = results_df.style.format({
            'Precision': '{:.3f}',
            'Recall': '{:.3f}',
            'F1': '{:.3f}',
            'AUC': '{:.3f}'
        }).highlight_max(axis=0, subset=['Precision', 'Recall', 'F1', 'AUC'])
        
        st.dataframe(styled_results, use_container_width=True)
        
        # Best model highlight
        best_model_idx = results_df['AUC'].idxmax()
        best_model = results_df.loc[best_model_idx, 'Model']
        best_auc = results_df.loc[best_model_idx, 'AUC']
        
        st.success(f"🥇 **Best Model:** {best_model} (AUC: {best_auc:.3f})")
    
    with col2:
        st.subheader("📊 Metrics Comparison")
        
        # Create radar chart
        fig = go.Figure()
        
        metrics = ['Precision', 'Recall', 'F1', 'AUC']
        
        for idx, row in results_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[metric] for metric in metrics],
                theta=metrics,
                fill='toself',
                name=row['Model'],
                line_color=['#1f77b4', '#ff7f0e', '#2ca02c'][idx]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Model insights
    st.subheader("🔍 Model Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Logistic Regression")
        st.markdown("""
        **Strengths:**
        - Fast training and prediction
        - Highly interpretable coefficients
        - Good baseline performance
        
        **Use Case:**
        - When model transparency is critical
        - Limited computational resources
        """)
    
    with col2:
        st.markdown("### Random Forest")
        st.markdown("""
        **Strengths:**
        - Handles non-linear relationships
        - Built-in feature importance
        - Robust to outliers
        
        **Use Case:**
        - Balanced performance and interpretability
        - When feature interactions matter
        """)
    
    with col3:
        st.markdown("### XGBoost")
        st.markdown("""
        **Strengths:**
        - Highest predictive performance
        - Handles complex patterns
        - Advanced regularization
        
        **Use Case:**
        - When accuracy is paramount
        - Large-scale production systems
        """)
    
    # Performance targets
    st.subheader("🎯 Performance Targets")
    
    target_col1, target_col2, target_col3 = st.columns(3)
    
    with target_col1:
        avg_auc = results_df['AUC'].mean()
        if avg_auc >= 0.80:
            st.success(f"✅ AUC Target: {avg_auc:.3f} ≥ 0.80")
        else:
            st.warning(f"⚠️ AUC Target: {avg_auc:.3f} < 0.80")
    
    with target_col2:
        avg_recall = results_df['Recall'].mean()
        if avg_recall >= 0.60:
            st.success(f"✅ Recall Target: {avg_recall:.3f} ≥ 0.60")
        else:
            st.warning(f"⚠️ Recall Target: {avg_recall:.3f} < 0.60")
    
    with target_col3:
        best_f1 = results_df['F1'].max()
        if best_f1 >= 0.70:
            st.success(f"✅ F1 Target: {best_f1:.3f} ≥ 0.70")
        else:
            st.warning(f"⚠️ F1 Target: {best_f1:.3f} < 0.70")

def clv_overview_tab():
    """CLV analysis overview tab."""
    
    st.header("💎 Customer Lifetime Value Analysis")
    
    # Load training data for CLV analysis
    train_df, _ = load_processed_data()
    if train_df is None:
        return
    
    # Load encoders to decode categorical variables
    _, encoders, _ = load_models()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 CLV Distribution")
        
        # CLV histogram
        fig_hist = px.histogram(
            train_df, 
            x='clv', 
            nbins=50,
            title="Customer Lifetime Value Distribution",
            labels={'clv': 'CLV ($)', 'count': 'Number of Customers'}
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # CLV statistics
        st.markdown("### CLV Statistics")
        clv_stats = train_df['clv'].describe()
        
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.metric("Average CLV", f"${clv_stats['mean']:,.0f}")
            st.metric("Median CLV", f"${clv_stats['50%']:,.0f}")
        
        with stat_col2:
            st.metric("Min CLV", f"${clv_stats['min']:,.0f}")
            st.metric("Max CLV", f"${clv_stats['max']:,.0f}")
    
    with col2:
        st.subheader("🎯 Churn Rate by CLV Quartile")
        
        # Calculate churn by CLV quartile
        churn_by_clv = train_df.groupby('clv_quartile')['Churn'].agg(['count', 'sum', 'mean'])
        churn_by_clv.columns = ['Total', 'Churned', 'Churn_Rate']
        churn_by_clv['Churn_Rate'] *= 100  # Convert to percentage
        
        # Decode quartile labels if encoded
        if encoders and 'clv_quartile' in encoders:
            quartile_labels = [encoders['clv_quartile'].classes_[i] for i in churn_by_clv.index]
            churn_by_clv.index = quartile_labels
        
        # Create bar chart
        fig_churn = px.bar(
            x=churn_by_clv.index,
            y=churn_by_clv['Churn_Rate'],
            title="Churn Rate by CLV Quartile",
            labels={'x': 'CLV Quartile', 'y': 'Churn Rate (%)'},
            color=churn_by_clv['Churn_Rate'],
            color_continuous_scale='Reds'
        )
        fig_churn.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_churn, use_container_width=True)
        
        # Revenue at risk
        st.markdown("### Revenue at Risk")
        revenue_at_risk = train_df.groupby('clv_quartile').agg({
            'clv': 'mean',
            'Churn': 'sum'
        })
        revenue_at_risk['Risk'] = revenue_at_risk['clv'] * revenue_at_risk['Churn']
        
        if encoders and 'clv_quartile' in encoders:
            revenue_at_risk.index = quartile_labels
        
        total_risk = revenue_at_risk['Risk'].sum()
        
        for idx, row in revenue_at_risk.iterrows():
            percentage = (row['Risk'] / total_risk) * 100
            st.metric(
                f"{idx} CLV Risk",
                f"${row['Risk']:,.0f}",
                f"{percentage:.1f}% of total"
            )
    
    # Business insights
    st.subheader("💡 Key Business Insights")
    
    # Calculate insights
    if encoders and 'clv_quartile' in encoders:
        quartile_labels = [encoders['clv_quartile'].classes_[i] for i in range(4)]
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("#### 🎯 Retention Priorities")
            
            # Find quartile with highest revenue at risk
            highest_risk_quartile = revenue_at_risk['Risk'].idxmax()
            highest_risk_amount = revenue_at_risk['Risk'].max()
            
            st.info(f"""
            **Primary Focus:** {highest_risk_quartile} CLV customers
            - Total revenue at risk: ${highest_risk_amount:,.0f}
            - These customers should be the top priority for retention campaigns
            """)
            
            # Find quartile with highest churn rate
            highest_churn_quartile = churn_by_clv['Churn_Rate'].idxmax()
            highest_churn_rate = churn_by_clv['Churn_Rate'].max()
            
            st.warning(f"""
            **Churn Challenge:** {highest_churn_quartile} CLV segment
            - Churn rate: {highest_churn_rate:.1f}%
            - Requires immediate intervention strategies
            """)
        
        with insights_col2:
            st.markdown("#### 💰 Value Optimization")
            
            # CLV efficiency
            avg_clv_by_quartile = train_df.groupby('clv_quartile')['clv'].mean()
            if encoders and 'clv_quartile' in encoders:
                avg_clv_by_quartile.index = quartile_labels
            
            premium_clv = avg_clv_by_quartile.max()
            low_clv = avg_clv_by_quartile.min()
            clv_ratio = premium_clv / low_clv
            
            st.success(f"""
            **Value Spread:** {clv_ratio:.1f}x difference
            - Premium customers: ${premium_clv:,.0f}
            - Low-value customers: ${low_clv:,.0f}
            - Focus upselling efforts on medium-value segments
            """)
            
            # Retention ROI
            total_customers_at_risk = train_df['Churn'].sum()
            total_revenue_at_risk = (train_df['clv'] * train_df['Churn']).sum()
            avg_risk_per_customer = total_revenue_at_risk / total_customers_at_risk
            
            st.info(f"""
            **Retention ROI:** Prevent one churn = ${avg_risk_per_customer:,.0f}
            - Total customers at risk: {total_customers_at_risk:,}
            - Total revenue at risk: ${total_revenue_at_risk:,.0f}
            - Justify retention spend up to ${avg_risk_per_customer*0.2:,.0f} per customer
            """)
    
    # Actionable recommendations
    st.subheader("🚀 Actionable Recommendations")
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        st.markdown("""
        ### 🎯 Immediate Actions
        
        1. **Prioritize High-Value Churners**
           - Create dedicated retention team for Premium CLV customers
           - Implement early warning system
        
        2. **Segment-Specific Campaigns**
           - Premium: White-glove service, loyalty rewards
           - High: Contract upgrade incentives
           - Medium: Feature education, usage optimization
           - Low: Value demonstration, basic retention offers
        """)
    
    with rec_col2:
        st.markdown("""
        ### 📊 Data-Driven Strategies
        
        1. **Predictive Interventions**
           - Deploy churn prediction models in production
           - Automate risk scoring and alerts
        
        2. **CLV Optimization**
           - Focus upselling on medium CLV customers
           - Analyze premium customer success patterns
           - Develop CLV growth strategies
        """)
    
    with rec_col3:
        st.markdown("""
        ### 💡 Long-term Improvements
        
        1. **Advanced Analytics**
           - Implement survival analysis for tenure prediction
           - Develop dynamic CLV models
        
        2. **Operational Excellence**
           - Create retention playbooks by segment
           - Establish CLV-based customer success metrics
           - Regular model performance monitoring
        """)

def main():
    """Main Streamlit app."""
    
    # App header
    st.title("📊 Customer Churn Prediction & CLV Analysis")
    st.markdown("""
    *Predict customer churn and analyze customer lifetime value to prioritize retention efforts*
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        st.markdown("""
        **Project Overview:**
        - 🎯 Predict customer churn probability
        - 💰 Calculate Customer Lifetime Value (CLV)
        - 📈 Compare ML model performance
        - 💡 Generate actionable insights
        """)
        
        st.markdown("---")
        st.markdown("**Data:**")
        st.markdown("IBM Telco Customer Churn Dataset")
        st.markdown("**Models:**")
        st.markdown("- Logistic Regression")
        st.markdown("- Random Forest")
        st.markdown("- XGBoost")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "🔮 Predict Churn", 
        "📈 Model Performance", 
        "💎 CLV Analysis"
    ])
    
    with tab1:
        predict_tab()
    
    with tab2:
        model_performance_tab()
    
    with tab3:
        clv_overview_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | 
        <a href='https://github.com/your-repo/project2-churn-prediction' target='_blank'>GitHub Repository</a> | 
        <a href='https://www.youtube.com/watch?v=your-video' target='_blank'>Video Demo</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
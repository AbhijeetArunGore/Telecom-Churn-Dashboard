import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Telecom Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📞 Telecom Customer Churn Analysis Dashboard")
st.markdown("Analyze customer behavior and predict churn probability")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\abhis\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # Preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df

df = load_data()

# Sidebar
st.sidebar.header("Dashboard Controls")
selected_analysis = st.sidebar.selectbox(
    "Select Analysis",
    ["Overview", "Churn Analysis", "Customer Segmentation", "Churn Prediction"]
)

# Main content based on selection
if selected_analysis == "Overview":
    st.header("📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(df))
    
    with col2:
        churn_rate = df['Churn'].mean()
        st.metric("Churn Rate", f"{churn_rate:.1%}")
    
    with col3:
        avg_tenure = df['tenure'].mean()
        st.metric("Avg Tenure", f"{avg_tenure:.1f} months")
    
    with col4:
        avg_monthly = df['MonthlyCharges'].mean()
        st.metric("Avg Monthly Charge", f"${avg_monthly:.2f}")
    
    # Show data sample
    st.subheader("Data Sample")
    st.dataframe(df.head(10))
    
    # Basic statistics
    st.subheader("Basic Statistics")
    st.dataframe(df[['tenure', 'MonthlyCharges', 'TotalCharges']].describe())

elif selected_analysis == "Churn Analysis":
    st.header("🔍 Churn Analysis")
    
    # Churn distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        churn_counts = df['Churn'].value_counts()
        ax.pie(churn_counts.values, labels=['Not Churned', 'Churned'], 
               autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        st.pyplot(fig)
    
    with col2:
        st.subheader("Churn by Contract Type")
        contract_churn = pd.crosstab(df['Contract'], df['Churn'])
        fig, ax = plt.subplots(figsize=(8, 6))
        contract_churn.plot(kind='bar', ax=ax, color=['lightblue', 'lightcoral'])
        ax.set_title('Churn by Contract Type')
        ax.legend(['No Churn', 'Churn'])
        st.pyplot(fig)
    
    # Tenure vs Churn
    st.subheader("Tenure Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Churn', y='tenure', data=df, ax=ax)
        ax.set_title('Tenure vs Churn')
        ax.set_xlabel('Churn (0=No, 1=Yes)')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=ax)
        ax.set_title('Monthly Charges vs Churn')
        ax.set_xlabel('Churn (0=No, 1=Yes)')
        st.pyplot(fig)

elif selected_analysis == "Customer Segmentation":
    st.header("👥 Customer Segmentation")
    
    # Segment by tenure
    st.subheader("Customer Segments by Tenure")
    
    tenure_bins = [0, 12, 24, 60, 100]
    tenure_labels = ['New (0-12m)', 'Regular (12-24m)', 'Loyal (24-60m)', 'VIP (60+m)']
    
    df['TenureSegment'] = pd.cut(df['tenure'], bins=tenure_bins, labels=tenure_labels)
    
    col1, col2 = st.columns(2)
    
    with col1:
        segment_counts = df['TenureSegment'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        segment_counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Customer Count by Tenure Segment')
        ax.set_ylabel('Number of Customers')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        segment_churn = df.groupby('TenureSegment')['Churn'].mean()
        fig, ax = plt.subplots(figsize=(8, 6))
        segment_churn.plot(kind='bar', ax=ax, color='lightcoral')
        ax.set_title('Churn Rate by Tenure Segment')
        ax.set_ylabel('Churn Rate')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Internet service analysis
    st.subheader("Internet Service Analysis")
    internet_churn = pd.crosstab(df['InternetService'], df['Churn'], normalize='index')
    fig, ax = plt.subplots(figsize=(10, 6))
    internet_churn.plot(kind='bar', ax=ax, color=['lightblue', 'lightcoral'])
    ax.set_title('Churn Rate by Internet Service')
    ax.set_ylabel('Proportion')
    ax.legend(['No Churn', 'Churn'])
    st.pyplot(fig)

else:  # Churn Prediction
    st.header("🎯 Churn Prediction")
    
    st.markdown("Predict whether a customer is likely to churn based on their profile")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.slider("Tenure (months)", 0, 100, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 0, 200, 65)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    with col2:
        payment_method = st.selectbox("Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
    
    # Calculate total charges
    total_charges = monthly_charges * tenure
    
    # Simple prediction logic (for demo purposes)
    def simple_churn_prediction(inputs):
        # This is a simplified scoring system for demonstration
        score = 0
        
        # Contract scoring
        if inputs['contract'] == "Month-to-month":
            score += 3
        elif inputs['contract'] == "One year":
            score += 1
        
        # Tenure scoring
        if inputs['tenure'] < 12:
            score += 2
        elif inputs['tenure'] < 24:
            score += 1
        
        # Monthly charges scoring
        if inputs['monthly_charges'] > 70:
            score += 2
        elif inputs['monthly_charges'] > 50:
            score += 1
        
        # Internet service scoring
        if inputs['internet_service'] == "Fiber optic":
            score += 1
        
        # Payment method scoring
        if inputs['payment_method'] == "Electronic check":
            score += 2
        
        # Convert score to probability (simplified)
        probability = min(score / 10, 0.95)
        
        return probability
    
    if st.button("Predict Churn"):
        inputs = {
            'tenure': tenure,
            'monthly_charges': monthly_charges,
            'contract': contract,
            'internet_service': internet_service,
            'payment_method': payment_method,
            'paperless_billing': paperless_billing,
            'senior_citizen': senior_citizen,
            'dependents': dependents
        }
        
        probability = simple_churn_prediction(inputs)
        
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Churn Probability", f"{probability:.1%}")
        
        with col2:
            if probability >= 0.7:
                risk = "🔴 High"
            elif probability >= 0.4:
                risk = "🟡 Medium"
            else:
                risk = "🟢 Low"
            st.metric("Risk Level", risk)
        
        with col3:
            prediction = "Yes" if probability > 0.5 else "No"
            st.metric("Likely to Churn", prediction)
        
        # Progress bar
        st.progress(probability)
        
        # Recommendations
        st.subheader("Recommendations")
        if probability > 0.7:
            st.error("""
            **Immediate Action Required!**
            - Offer loyalty discount
            - Personal retention call
            - Special offer on contract renewal
            """)
        elif probability > 0.4:
            st.warning("""
            **Monitor Closely**
            - Send satisfaction survey
            - Offer service upgrade
            - Check for service issues
            """)
        else:
            st.success("""
            **Low Risk**
            - Continue standard service
            - Regular check-ins
            - Cross-sell additional services
            """)

st.markdown("---")
st.markdown("Built with Streamlit • Simple Telecom Churn Analysis Dashboard")

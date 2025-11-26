import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv(r"C:\Users\abhis\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nBasic Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())
# Create a copy
data = df.copy()

# Handle TotalCharges - convert to numeric and fill missing values
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(0, inplace=True)

# Convert Churn to binary (1 for Yes, 0 for No)
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# Drop customerID as it's not useful for modeling
data = data.drop('customerID', axis=1)

print("After preprocessing:")
print(f"Data shape: {data.shape}")
print(f"Churn distribution: {data['Churn'].value_counts()}")
print(f"Churn rate: {data['Churn'].mean():.2%}")
# Set up the plotting style
plt.style.use('default')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Churn Distribution
churn_counts = data['Churn'].value_counts()
axes[0,0].pie(churn_counts.values, labels=['Not Churned', 'Churned'], autopct='%1.1f%%', 
              colors=['lightblue', 'lightcoral'], startangle=90)
axes[0,0].set_title('Customer Churn Distribution')

# 2. Churn vs Tenure
sns.boxplot(x='Churn', y='tenure', data=data, ax=axes[0,1])
axes[0,1].set_title('Churn vs Tenure')
axes[0,1].set_xlabel('Churn (0=No, 1=Yes)')
axes[0,1].set_ylabel('Tenure (months)')

# 3. Churn vs Contract
contract_churn = pd.crosstab(data['Contract'], data['Churn'])
contract_churn_percentage = contract_churn.div(contract_churn.sum(axis=1), axis=0)
contract_churn_percentage.plot(kind='bar', ax=axes[0,2], color=['lightblue', 'lightcoral'])
axes[0,2].set_title('Churn Rate by Contract Type')
axes[0,2].set_xlabel('Contract Type')
axes[0,2].set_ylabel('Proportion')
axes[0,2].legend(['No Churn', 'Churn'])

# 4. Churn vs Monthly Charges
sns.boxplot(x='Churn', y='MonthlyCharges', data=data, ax=axes[1,0])
axes[1,0].set_title('Churn vs Monthly Charges')
axes[1,0].set_xlabel('Churn (0=No, 1=Yes)')
axes[1,0].set_ylabel('Monthly Charges ($)')

# 5. Churn vs Internet Service
internet_churn = pd.crosstab(data['InternetService'], data['Churn'])
internet_churn_percentage = internet_churn.div(internet_churn.sum(axis=1), axis=0)
internet_churn_percentage.plot(kind='bar', ax=axes[1,1], color=['lightblue', 'lightcoral'])
axes[1,1].set_title('Churn Rate by Internet Service')
axes[1,1].set_xlabel('Internet Service')
axes[1,1].set_ylabel('Proportion')
axes[1,1].legend(['No Churn', 'Churn'])

# 6. Churn by Senior Citizen
senior_churn = pd.crosstab(data['SeniorCitizen'], data['Churn'])
senior_churn_percentage = senior_churn.div(senior_churn.sum(axis=1), axis=0)
senior_churn_percentage.plot(kind='bar', ax=axes[1,2], color=['lightblue', 'lightcoral'])
axes[1,2].set_title('Churn Rate by Senior Citizen')
axes[1,2].set_xlabel('Senior Citizen (0=No, 1=Yes)')
axes[1,2].set_ylabel('Proportion')
axes[1,2].legend(['No Churn', 'Churn'])

plt.tight_layout()
plt.show()

# Print key insights
print("KEY INSIGHTS FROM EDA:")
print(f"1. Overall churn rate: {data['Churn'].mean():.1%}")
print(f"2. Average tenure for churned customers: {data[data['Churn']==1]['tenure'].mean():.1f} months")
print(f"3. Average tenure for non-churned: {data[data['Churn']==0]['tenure'].mean():.1f} months")
print(f"4. Monthly charges for churned: ${data[data['Churn']==1]['MonthlyCharges'].mean():.2f}")
print(f"5. Monthly charges for non-churned: ${data[data['Churn']==0]['MonthlyCharges'].mean():.2f}")
# Select only important features for our simple model
features_to_use = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
    'Contract', 'InternetService', 'PaymentMethod', 'PaperlessBilling'
]

# Create a new dataframe with selected features
simple_data = data[features_to_use + ['Churn']].copy()

# Create one simple engineered feature
simple_data['TenureGroup'] = pd.cut(simple_data['tenure'], 
                                   bins=[0, 12, 24, 60, 100], 
                                   labels=['New', 'Regular', 'Loyal', 'VIP'])

print("Simple dataset shape:", simple_data.shape)
print("\nFeatures used:", features_to_use)
print("New feature created: TenureGroup")
from sklearn.preprocessing import LabelEncoder

# Prepare data for modeling
model_data = simple_data.copy()

# Convert categorical variables to numerical using LabelEncoder
categorical_columns = ['Contract', 'InternetService', 'PaymentMethod', 'PaperlessBilling', 'TenureGroup']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    model_data[col] = le.fit_transform(model_data[col].astype(str))
    label_encoders[col] = le

# Separate features and target
X = model_data.drop('Churn', axis=1)
y = model_data['Churn']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Churn rate in training: {y_train.mean():.2%}")
print(f"Churn rate in test: {y_test.mean():.2%}")
# Train a Random Forest model (good for beginners)
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("MODEL PERFORMANCE:")
print("=" * 50)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Churn', 'Churn'], 
            yticklabels=['Not Churn', 'Churn'])
plt.title('Confusion Matrix - Customer Churn Prediction')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

print("\nTop 5 Most Important Features:")
for i, row in feature_importance.head().iterrows():
    print(f"{row['feature']}: {row['importance']:.3f}")

def predict_churn_simple(customer_data):
    """
    Predict churn for a single customer
    customer_data should be a dictionary with the features
    """
    # Create a dataframe from the input
    input_df = pd.DataFrame([customer_data])
    
    # Encode categorical variables
    for col in categorical_columns:
        if col in input_df.columns:
            # Handle unseen labels
            try:
                input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
            except ValueError:
                # If unseen label, use the most common one
                input_df[col] = 0
    
    # Make sure all columns are present
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training data
    input_df = input_df[X.columns]
    
    # Make prediction
    probability = model.predict_proba(input_df)[0, 1]
    prediction = model.predict(input_df)[0]
    
    # Risk category
    if probability >= 0.7:
        risk = "High"
    elif probability >= 0.4:
        risk = "Medium"
    else:
        risk = "Low"
    
    return {
        'churn_probability': round(probability, 3),
        'will_churn': bool(prediction),
        'risk_category': risk,
        'confidence': round(max(probability, 1-probability), 3)
    }

# Test with a sample customer
sample_customer = {
    'tenure': 5,
    'MonthlyCharges': 85.0,
    'TotalCharges': 425.0,
    'SeniorCitizen': 0,
    'Contract': 'Month-to-month',
    'InternetService': 'Fiber optic',
    'PaymentMethod': 'Electronic check',
    'PaperlessBilling': 'Yes',
    'TenureGroup': 'New'
}

result = predict_churn_simple(sample_customer)
print("PREDICTION RESULT:")
for key, value in result.items():
    print(f"{key}: {value}")


📊 Telecom Churn Analysis & Prediction Dashboard

A complete end-to-end Telecom Customer Churn project that includes:
Exploratory Data Analysis (EDA)
Data pre-processing
Machine Learning model for churn prediction
Fully interactive Streamlit dashboard with insights, charts, and prediction interface
This project helps telecom companies identify customers likely to churn so they can take proactive actions to retain them.

1. Project Overview

Customer churn is one of the biggest revenue killers in the telecom industry.
This project analyzes customer behavior, identifies churn patterns, and predicts high-risk customers using ML.
The interactive dashboard allows business teams to:
Visualize customer behavior
Understand churn drivers
Predict churn probability
Improve business decisions & reduce losses

2. Folder Structure
Telecom-Churn-Dashboard/
│
├── data/
│   └── telecom_churn.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── models/
│   └── churn_model.pkl
│
├── app.py                         
├── requirements.txt
├── README.md
└── utils.py                       

3. Features
Exploratory Data Analysis
- Customer demographics
- Service usage patterns
- Billing behavior
- Churn distribution
- Heatmaps, bar charts, pair plots

Machine Learning
- Preprocessing (scaling, encoding)
- Feature selection
- Model training (Logistic Regression / RandomForest / XGBoost)
- Evaluation metrics (accuracy, precision, recall, F1)
- Final model pickled for reuse

Dashboard (Streamlit)
- Clean, interactive UI
- Visual data insights
- Feature sliders/inputs
- On-the-spot churn prediction
- Real-time probability output

4. How the ML Model Works
- Convert raw customer info into numerical features
- Apply scaling + one-hot encoding
- Train ML algorithms
- Compare model performance
- Save best model (.pkl)
- Dashboard loads model and predicts churn for any input customer

5. Business Impact: How This Reduces Churn
- Identifies high-risk customers early
- Helps teams launch retention campaigns
- Prioritizes users with higher churn probability
- Improves customer lifetime value
- Cuts customer acquisition costs by retaining existing users

6. How to Run the Project
Step 1 — Clone Repo
git clone https://github.com/AbhijeetArunGore/Telecom-Churn-Dashboard
cd Telecom-Churn-Dashboard

Step 2 — Install Dependencies
pip install -r requirements.txt

Step 3 — Run Streamlit Dashboard
streamlit run app.py

Dashboard will open at:
http://localhost:8501/

7. Requirements
- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly / Matplotlib
- Joblib
(All installed via requirements.txt.)

8. Dataset Used
Dataset: Telecom Customer Churn Dataset
Contains fields like:
- Customer ID
- Gender
- Tenure
- Monthly charges
- Total charges
- Internet services
- Contract type
- Payment method
- Churn flag
Your repo’s data/ folder includes the dataset.

9. Outputs
Visual Insights
- Churn vs non-churn distribution
- Tenure-based churn
- Billing & usage patterns

Model Outputs
- Accuracy score
- Confusion matrix
- Feature importance

Dashboard Output
- Churn probability (%)
- Clear “Likely to Churn / Not Likely” label

10. Author
Abhijeet Arun Gore
Data Science Enthusiast | ML Engineer
GitHub: https://github.com/AbhijeetArunGore

LinkedIn: https://www.linkedin.com/in/abhijeet-gore-972225282/

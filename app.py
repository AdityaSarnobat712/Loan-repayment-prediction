import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import shap
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Loan Repayment Prediction", layout="wide")

# Title and description
st.title("Loan Repayment Prediction System")
st.write("Enter customer details to predict loan repayment likelihood using tree-based models.")

# Load saved models and scaler
@st.cache_resource
def load_models():
    xgb_model = joblib.load('models/xgboost_model.pkl')
    rf_model = joblib.load('models/random_forest_model.pkl')
    dt_model = joblib.load('models/decision_tree_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return xgb_model, rf_model, dt_model, scaler

xgb_model, rf_model, dt_model, scaler = load_models()

# Define feature columns (based on dataset, excluding Customer_ID and Delinquent_Account)
feature_columns = [
    'Age', 'Income', 'Credit_Score', 'Credit_Utilization', 'Missed_Payments',
    'Loan_Balance', 'Debt_to_Income_Ratio', 'Employment_Status', 'Account_Tenure',
    'Credit_Card_Type', 'Location', 'Month_1', 'Month_2', 'Month_3', 'Month_4',
    'Month_5', 'Month_6', 'Payment_Score'
]

# Sidebar for model selection
st.sidebar.header("Select Model")
model_choice = st.sidebar.selectbox("Choose a model for prediction:", 
                                   ["XGBoost", "Random Forest", "Decision Tree", "All Models"])

# Sidebar for user input
st.sidebar.header("Customer Information")

# Numerical inputs
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
income = st.sidebar.number_input("Income ($)", min_value=0.0, value=50000.0)
credit_score = st.sidebar.number_input("Credit Score", min_value=300.0, max_value=850.0, value=600.0)
credit_utilization = st.sidebar.number_input("Credit Utilization Ratio", min_value=0.0, max_value=1.0, value=0.3)
missed_payments = st.sidebar.number_input("Missed Payments", min_value=0, value=0)
loan_balance = st.sidebar.number_input("Loan Balance ($)", min_value=0.0, value=10000.0)
debt_to_income = st.sidebar.number_input("Debt to Income Ratio", min_value=0.0, max_value=1.0, value=0.3)
account_tenure = st.sidebar.number_input("Account Tenure (months)", min_value=0, value=12)

# Categorical inputs
employment_status = st.sidebar.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed"])
credit_card_type = st.sidebar.selectbox("Credit Card Type", ["Standard", "Platinum", "Student"])
location = st.sidebar.selectbox("Location", ["Los Angeles", "Phoenix", "Chicago"])

# Payment history inputs
payment_mapping = {'On-time': 0, 'Late': 1, 'Missed': 2}
month_1 = st.sidebar.selectbox("Month 1 Payment Status", ["On-time", "Late", "Missed"])
month_2 = st.sidebar.selectbox("Month 2 Payment Status", ["On-time", "Late", "Missed"])
month_3 = st.sidebar.selectbox("Month 3 Payment Status", ["On-time", "Late", "Missed"])
month_4 = st.sidebar.selectbox("Month 4 Payment Status", ["On-time", "Late", "Missed"])
month_5 = st.sidebar.selectbox("Month 5 Payment Status", ["On-time", "Late", "Missed"])
month_6 = st.sidebar.selectbox("Month 6 Payment Status", ["On-time", "Late", "Missed"])

# Preprocess input data
def preprocess_input():
    input_data = {
        'Age': age,
        'Income': income,
        'Credit_Score': credit_score,
        'Credit_Utilization': credit_utilization,
        'Missed_Payments': missed_payments,
        'Loan_Balance': loan_balance,
        'Debt_to_Income_Ratio': debt_to_income,
        'Employment_Status': employment_status,
        'Account_Tenure': account_tenure,
        'Credit_Card_Type': credit_card_type,
        'Location': location,
        'Month_1': payment_mapping[month_1],
        'Month_2': payment_mapping[month_2],
        'Month_3': payment_mapping[month_3],
        'Month_4': payment_mapping[month_4],
        'Month_5': payment_mapping[month_5],
        'Month_6': payment_mapping[month_6],
        'Payment_Score': sum([payment_mapping[month_1], payment_mapping[month_2], 
                              payment_mapping[month_3], payment_mapping[month_4], 
                              payment_mapping[month_5], payment_mapping[month_6]])
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    employment_map = {'Employed': 0, 'Self-employed': 2, 'Unemployed': 3}
    credit_card_map = {'Standard': 3, 'Platinum': 2, 'Student': 4}
    location_map = {'Los Angeles': 2, 'Phoenix': 4, 'Chicago': 0}
    
    input_df['Employment_Status'] = input_df['Employment_Status'].map(employment_map)
    input_df['Credit_Card_Type'] = input_df['Credit_Card_Type'].map(credit_card_map)
    input_df['Location'] = input_df['Location'].map(location_map)
    
    if input_df.isnull().sum().sum() > 0:
        st.error("Input data contains missing values. Please check your inputs.")
        return None, None
    
    input_scaled = scaler.transform(input_df[feature_columns])
    return input_scaled, input_df

# Function to get SHAP explanations
def get_shap_explanation(model, input_scaled):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)
    if isinstance(shap_values, list):  # XGBoost may return a list
        shap_values = shap_values[1]
    return explainer, shap_values

# Function to plot SHAP summary with reduced size
def plot_shap_summary(explainer, shap_values, input_df, model_name):
    plt.figure(figsize=(6, 4))  # Smaller figure size
    shap.summary_plot(shap_values, input_df, feature_names=feature_columns, show=False)
    plt.tight_layout()  # Ensure layout fits
    st.pyplot(plt, bbox_inches='tight')

# Function to get feature importance
def get_feature_importance(model):
    importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    return importance

# Predict and display results
if st.sidebar.button("Predict"):
    input_scaled, input_df = preprocess_input()
    
    if input_scaled is None:
        st.stop()
    
    models = {
        'XGBoost': xgb_model,
        'Random Forest': rf_model,
        'Decision Tree': dt_model
    }
    
    predictions = {}
    probabilities = {}
    recommendations = {}
    
    for name, model in models.items():
        pred = model.predict(input_scaled)[0]
        pred_proba = model.predict_proba(input_scaled)[0][0]  # Probability of class 0 (non-delinquent)
        predictions[name] = "Will Repay" if pred == 0 else "Will Not Repay"
        probabilities[name] = pred_proba
        recommendations[name] = "Approve Loan" if pred_proba >= 0.6 else "Deny Loan"
    
    st.header("Prediction Results")
    if model_choice == "All Models":
        # Use columns to display results compactly
        cols = st.columns(3)  # One column per model
        for idx, name in enumerate(models.keys()):
            with cols[idx]:
                st.subheader(f"{name} Model")
                st.write(f"Prediction: {predictions[name]}")
                st.write(f"Repayment Probability: {probabilities[name]*100:.2f}%")
                st.write(f"Recommendation: {recommendations[name]}")
                
                st.write(f"SHAP Explanation for {name}")
                explainer, shap_values = get_shap_explanation(model, input_scaled)
                plot_shap_summary(explainer, shap_values, input_df, name)
                
                st.write(f"Feature Importance for {name}")
                importance = get_feature_importance(model)
                st.table(importance)
    else:
        st.subheader(f"{model_choice} Model")
        st.write(f"Prediction: {predictions[model_choice]}")
        st.write(f"Repayment Probability: {probabilities[model_choice]*100:.2f}%")
        st.write(f"Recommendation: {recommendations[model_choice]}")
        
        st.write(f"SHAP Explanation for {model_choice}")
        explainer, shap_values = get_shap_explanation(models[model_choice], input_scaled)
        plot_shap_summary(explainer, shap_values, input_df, model_choice)
        
        st.write(f"Feature Importance for {model_choice}")
        importance = get_feature_importance(models[model_choice])
        st.table(importance)
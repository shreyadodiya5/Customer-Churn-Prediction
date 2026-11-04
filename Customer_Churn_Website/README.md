# Customer Churn Prediction Web App

This Streamlit website is based on `Customer_Churn_Project.ipynb`.

## Features
- Overview dashboard and churn KPIs
- Exploratory charts for contract, tenure and internet service
- Single-customer churn prediction using Logistic Regression
- Bulk CSV predictions with downloadable results
- Logistic Regression vs Random Forest evaluation
- Confusion matrix and Random Forest feature importance

## Run locally (Windows)
1. Install Python 3.10 or newer.
2. Extract this folder.
3. Open Command Prompt in this folder.
4. Run:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Dataset and training
The app downloads the same Telco Customer Churn CSV source used in the notebook when it starts, cleans `TotalCharges`, and trains the two pipelines using an 80/20 stratified split with `random_state=42`. Internet access is required on first run. The Logistic Regression pipeline is saved locally as `best_churn_model.joblib`.

## Bulk CSV format
Upload a CSV with the 19 input feature columns used in the notebook. Do not include `customerID`; the target `Churn` column is not required. Use the in-app sample CSV as a template.

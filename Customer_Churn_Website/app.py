import io
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Customer Churn Analytics", page_icon="📊", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/aiplanethub/Datasets/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_FILE = "best_churn_model.joblib"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()
    return df

@st.cache_resource
def train_models():
    df = load_data()
    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"].map({"No": 0, "Yes": 1})
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    prep = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    lr = Pipeline([("preprocessor", prep), ("classifier", LogisticRegression(max_iter=1000))])
    rf = Pipeline([("preprocessor", prep), ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))])
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    models = {"Logistic Regression": lr, "Random Forest": rf}
    metrics = []
    for name, model in models.items():
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob)
        })
    return df, X, y, X_test, y_test, models, pd.DataFrame(metrics)

try:
    df, X, y, X_test, y_test, models, metrics = train_models()
except Exception as e:
    st.error("Could not load the training dataset or train the models. Check your internet connection and try again.")
    st.exception(e)
    st.stop()

with open(MODEL_FILE, "wb") as f:
    joblib.dump({"pipeline": models["Logistic Regression"],
                 "config": {"target_col": "Churn", "positive_class": "Yes",
                            "exclude_cols": ["customerID"], "test_size": 0.2,
                            "random_state": 42, "model_save_path": MODEL_FILE},
                 "numeric_features": X.select_dtypes(include=["int64", "float64"]).columns.tolist(),
                 "categorical_features": X.select_dtypes(include=["object"]).columns.tolist()}, f)

st.sidebar.title("📊 Churn Analytics")
page = st.sidebar.radio("Navigation", ["Overview", "Explore Data", "Single Prediction", "Bulk Prediction", "Model Performance"])
st.sidebar.caption("Telecom customer churn prediction")

if page == "Overview":
    st.title("Customer Churn Prediction")
    st.markdown("Understand customer behavior, explore churn patterns, and predict which customers may leave.")
    total = len(df)
    churned = int((df.Churn == "Yes").sum())
    rate = churned / total * 100
    a,b,c,d = st.columns(4)
    a.metric("Total Customers", f"{total:,}")
    b.metric("Churned Customers", f"{churned:,}")
    c.metric("Churn Rate", f"{rate:.1f}%")
    d.metric("Features Used", f"{X.shape[1]}")
    left,right = st.columns(2)
    with left:
        st.subheader("Customer Churn Distribution")
        counts = df["Churn"].value_counts().rename_axis("Status").reset_index(name="Customers")
        st.plotly_chart(px.pie(counts, names="Status", values="Customers", hole=.55), use_container_width=True)
    with right:
        st.subheader("Churn by Contract Type")
        contract = (pd.crosstab(df.Contract, df.Churn, normalize="index") * 100).reset_index().melt("Contract", var_name="Status", value_name="Percent")
        st.plotly_chart(px.bar(contract, x="Contract", y="Percent", color="Status", barmode="group"), use_container_width=True)
    st.info("Models are trained from the Telco Customer Churn dataset using the same 80/20 stratified split and preprocessing approach as the notebook.")

elif page == "Explore Data":
    st.title("Explore Customer Data")
    st.write("Dataset preview")
    st.dataframe(df.head(100), use_container_width=True)
    l,r = st.columns(2)
    with l:
        st.subheader("Churn by Customer Tenure")
        temp = df.copy()
        temp["Tenure Group"] = pd.cut(temp.tenure, bins=[0,12,24,48,72], labels=["0–12","13–24","25–48","49–72"])
        chart = (pd.crosstab(temp["Tenure Group"], temp.Churn, normalize="index")*100).reset_index().melt("Tenure Group", var_name="Status", value_name="Percent")
        st.plotly_chart(px.bar(chart, x="Tenure Group", y="Percent", color="Status", barmode="group"), use_container_width=True)
    with r:
        st.subheader("Churn by Internet Service")
        chart = (pd.crosstab(df.InternetService, df.Churn, normalize="index")*100).reset_index().melt("InternetService", var_name="Status", value_name="Percent")
        st.plotly_chart(px.bar(chart, x="InternetService", y="Percent", color="Status", barmode="group"), use_container_width=True)
    st.subheader("Dataset summary")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

elif page == "Single Prediction":
    st.title("Predict Churn for One Customer")
    st.write("Enter customer details. The Logistic Regression pipeline from the notebook will estimate churn probability.")
    with st.form("customer_form"):
        c1,c2,c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Female","Male"])
            senior = st.selectbox("Senior Citizen", [0,1])
            partner = st.selectbox("Partner", ["Yes","No"])
            dependents = st.selectbox("Dependents", ["Yes","No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone Service", ["Yes","No"])
            multiple = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])
        with c2:
            internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
            security = st.selectbox("Online Security", ["Yes","No","No internet service"])
            backup = st.selectbox("Online Backup", ["Yes","No","No internet service"])
            protection = st.selectbox("Device Protection", ["Yes","No","No internet service"])
            support = st.selectbox("Tech Support", ["Yes","No","No internet service"])
            tv = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
            movies = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])
        with c3:
            contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes","No"])
            payment = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
            monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=1.0)
            total = st.number_input("Total Charges", min_value=0.0, value=float(tenure*monthly), step=1.0)
        submitted = st.form_submit_button("Predict Customer Churn", type="primary", use_container_width=True)
    if submitted:
        customer = pd.DataFrame([{
            "gender":gender, "SeniorCitizen":int(senior), "Partner":partner, "Dependents":dependents,
            "tenure":int(tenure), "PhoneService":phone, "MultipleLines":multiple,
            "InternetService":internet, "OnlineSecurity":security, "OnlineBackup":backup,
            "DeviceProtection":protection, "TechSupport":support, "StreamingTV":tv,
            "StreamingMovies":movies, "Contract":contract, "PaperlessBilling":paperless,
            "PaymentMethod":payment, "MonthlyCharges":float(monthly), "TotalCharges":float(total)
        }])
        model = models["Logistic Regression"]
        pred = int(model.predict(customer)[0])
        probability = float(model.predict_proba(customer)[0][1])*100
        x,yc = st.columns(2)
        x.metric("Predicted Status", "Likely to Churn" if pred == 1 else "Likely to Stay")
        yc.metric("Estimated Churn Probability", f"{probability:.2f}%")
        st.progress(min(max(probability/100,0),1))
        if pred == 1:
            st.warning("This customer is predicted to churn. Consider reviewing service experience, contract type, and retention offers.")
        else:
            st.success("This customer is predicted not to churn based on the model.")
        st.caption("This is a model estimate based on historical patterns, not a guarantee of customer behavior.")

elif page == "Bulk Prediction":
    st.title("Bulk Customer Churn Prediction")
    st.write("Upload a CSV containing customer feature columns. The target column `Churn` is not required.")
    st.download_button("Download sample input CSV", data=X.head(5).to_csv(index=False), file_name="sample_customer_input.csv", mime="text/csv")
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded:
        incoming = pd.read_csv(uploaded)
        expected = X.columns.tolist()
        missing = [col for col in expected if col not in incoming.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            clean = incoming[expected].copy()
            if "TotalCharges" in clean:
                clean["TotalCharges"] = pd.to_numeric(clean["TotalCharges"], errors="coerce").fillna(0)
            pred = models["Logistic Regression"].predict(clean)
            prob = models["Logistic Regression"].predict_proba(clean)[:,1]
            result = incoming.copy()
            result["Predicted_Churn"] = np.where(pred == 1, "Yes", "No")
            result["Churn_Probability_Pct"] = np.round(prob*100,2)
            st.success(f"Generated predictions for {len(result):,} customers.")
            st.dataframe(result, use_container_width=True)
            st.download_button("Download prediction results", data=result.to_csv(index=False).encode("utf-8"), file_name="customer_churn_predictions.csv", mime="text/csv")

elif page == "Model Performance":
    st.title("Model Performance")
    st.write("Evaluation on the notebook's stratified 20% test split.")
    st.dataframe(metrics.style.format({c:"{:.3f}" for c in metrics.columns if c!="Model"}), use_container_width=True)
    st.subheader("Compare model metrics")
    long = metrics.melt("Model", var_name="Metric", value_name="Score")
    st.plotly_chart(px.bar(long, x="Metric", y="Score", color="Model", barmode="group", range_y=[0,1]), use_container_width=True)
    selected = st.selectbox("Confusion matrix model", list(models.keys()))
    pred = models[selected].predict(X_test)
    cm = confusion_matrix(y_test, pred)
    fig = px.imshow(cm, text_auto=True, x=["Predicted No","Predicted Yes"], y=["Actual No","Actual Yes"], color_continuous_scale="Blues", aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
    rf = models["Random Forest"]
    names = rf.named_steps["preprocessor"].get_feature_names_out()
    imp = rf.named_steps["classifier"].feature_importances_
    importance = pd.DataFrame({"Feature":names,"Importance":imp}).nlargest(10,"Importance").sort_values("Importance")
    st.subheader("Top 10 Feature Importances — Random Forest")
    st.plotly_chart(px.bar(importance, x="Importance", y="Feature", orientation="h"), use_container_width=True)

st.sidebar.divider()
st.sidebar.caption("Customer Churn Prediction • Machine Learning Project")

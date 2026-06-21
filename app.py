import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Customer Churn Diagnosis", page_icon="📊", layout="wide")


# 1. Load data and train model (cached so it only runs once per session)

@st.cache_resource
def load_and_train():
    df = pd.read_csv("telco_churn_raw.csv")

    # --- Cleaning (same logic as notebook 01) ---
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # --- Prepare features (same logic as notebook 03) ---
    model_df = df.drop("customerID", axis=1)
    model_df["Churn"] = model_df["Churn"].map({"No": 0, "Yes": 1})

    cat_cols = model_df.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col])
        encoders[col] = le

    X = model_df.drop("Churn", axis=1)
    y = model_df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    return rf, encoders, X.columns.tolist(), df


rf_model, encoders, feature_order, raw_df = load_and_train()


# 2. Sidebar — customer input form

st.title("📊 Customer Churn Diagnosis & Retention Tool")
st.caption(
    "Predicts churn risk for a telecom customer and recommends a retention action, "
    "based on a Random Forest model trained on the IBM Telco Customer Churn dataset."
)

st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)

phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox(
    "Multiple Lines", ["No phone service", "No", "Yes"]
)
internet_service = st.sidebar.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)

def internet_dependent_options():
    if internet_service == "No":
        return ["No internet service"]
    return ["No", "Yes"]

online_security = st.sidebar.selectbox("Online Security", internet_dependent_options())
online_backup = st.sidebar.selectbox("Online Backup", internet_dependent_options())
device_protection = st.sidebar.selectbox("Device Protection", internet_dependent_options())
tech_support = st.sidebar.selectbox("Tech Support", internet_dependent_options())
streaming_tv = st.sidebar.selectbox("Streaming TV", internet_dependent_options())
streaming_movies = st.sidebar.selectbox("Streaming Movies", internet_dependent_options())

contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
total_charges = monthly_charges * tenure

predict_btn = st.sidebar.button("🔍 Predict Churn Risk", type="primary", use_container_width=True)


# 3. Build input row, encode, predict

def build_input_row():
    row = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    input_df = pd.DataFrame([row])

    for col, le in encoders.items():
        if col in input_df.columns:
            # Handle any category not seen during training gracefully
            val = input_df.at[0, col]
            if val in le.classes_:
                input_df[col] = le.transform([val])
            else:
                input_df[col] = 0

    return input_df[feature_order]


RETENTION_COST = 500
ASSUMED_REMAINING_MONTHS = 24


def segment_customer(prob, clv, value_threshold):
    high_risk = prob >= 0.5
    high_value = clv >= value_threshold
    if high_risk and high_value:
        return "🔴 Call Immediately (High Risk, High Value)", "#E63946"
    elif high_risk and not high_value:
        return "🟠 Low-Cost Offer (High Risk, Low Value)", "#F4A261"
    elif not high_risk and high_value:
        return "🟢 Monitor (Low Risk, High Value)", "#2A9D8F"
    else:
        return "⚪ No Action (Low Risk, Low Value)", "#A8DADC"


# Reference median CLV across the dataset for value threshold
reference_clv_median = (raw_df["MonthlyCharges"] * ASSUMED_REMAINING_MONTHS).median()

col1, col2 = st.columns([1, 1])

if predict_btn:
    input_row = build_input_row()
    prob = rf_model.predict_proba(input_row)[0, 1]
    clv = monthly_charges * ASSUMED_REMAINING_MONTHS
    expected_value = prob * clv
    net_value = expected_value - RETENTION_COST
    segment_label, segment_color = segment_customer(prob, clv, reference_clv_median)

    with col1:
        st.subheader("Prediction Result")
        st.metric("Churn Probability", f"{prob*100:.1f}%")
        st.progress(min(int(prob * 100), 100))

        if prob >= 0.5:
            st.error("⚠️ This customer is at HIGH risk of churning")
        else:
            st.success("✅ This customer is at LOW risk of churning")

        st.markdown(f"### Recommended Action")
        st.markdown(
            f"<div style='padding:12px;border-radius:8px;background-color:{segment_color}22;"
            f"border-left:6px solid {segment_color};font-size:18px;'>{segment_label}</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.subheader("Business Impact Estimate")
        st.metric("Estimated Customer Lifetime Value (CLV)", f"₹{clv:,.0f}")
        st.metric("Expected Value If Retained", f"₹{expected_value:,.0f}")
        worth_calling = net_value > 0 and prob >= 0.5
        st.metric(
            "Net Value of Intervention",
            f"₹{net_value:,.0f}",
            delta=f"{'Worth calling' if worth_calling else 'Not a priority right now'}",
        )
        st.caption(
            f"Assumptions: retention cost = ₹{RETENTION_COST}/customer, "
            f"CLV = Monthly Charges × {ASSUMED_REMAINING_MONTHS} months."
        )
else:
    st.info("👈 Enter customer details in the sidebar and click **Predict Churn Risk** to see results.")

st.divider()
st.caption(
    "Model: Random Forest Classifier (200 trees, max depth 10, class-weight balanced) | "
    "Trained on IBM Telco Customer Churn dataset (7,043 customers) | "
    "Test set performance: 70% recall on churn class, ROC-AUC 0.835"
)
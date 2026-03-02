import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Artifacts
log_model = joblib.load("logistic_model_balanced.pkl")
tree_model = joblib.load("decision_tree_balanced.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

df = pd.read_csv("cleaned_vehicle_data.csv")

st.set_page_config(page_title="Vehicle Maintenance Risk System", layout="wide")

st.title("Vehicle Maintenance Risk Assessment System")
st.write("Predict whether a vehicle requires maintenance using Logistic Regression or Decision Tree.")

# Model Selection
model_choice = st.radio("Select Model", ["Logistic Regression", "Decision Tree (Prefered Model)"])

# User Inputs (Must Match Training)
st.header("Enter Vehicle Details")

col1, col2 = st.columns(2)

with col1:
    mileage = st.number_input("Mileage", min_value=0.0, value=40000.0)
    vehicle_age = st.number_input("Vehicle Age (Years)", min_value=0.0, value=4.0)
    reported_issues = st.number_input("Reported Issues", min_value=0.0, value=2.0)

with col2:
    engine_size = st.number_input("Engine Size (cc)", min_value=0.0, value=1500.0)
    service_history = st.number_input("Service History Count", min_value=0.0, value=6.0)
    odometer = st.number_input("Odometer Reading", min_value=0.0, value=45000.0)

# Build Input Row
input_dict = {}

for col in df.columns:
    if col == "Need_Maintenance":
        continue

    if df[col].dtype == "object":
        input_dict[col] = df[col].mode()[0]
    else:
        input_dict[col] = df[col].median()

# Override with user inputs
input_dict["Mileage"] = mileage
input_dict["Vehicle_Age"] = vehicle_age
input_dict["Reported_Issues"] = reported_issues
input_dict["Engine_Size"] = engine_size
input_dict["Service_History"] = service_history
input_dict["Odometer_Reading"] = odometer

# Feature engineering (MUST match training)
input_dict["mileage_per_year"] = mileage / (vehicle_age + 1)

input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df)

# Align with training columns
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_columns]

# Prediction
if st.button("Predict Maintenance Requirement"):

    threshold = 0.7

    if model_choice == "Logistic Regression":
        input_scaled = scaler.transform(input_df)
        probability = log_model.predict_proba(input_scaled)[0][1]
    else:
        probability = tree_model.predict_proba(input_df)[0][1]

    prediction = 1 if probability > threshold else 0

    st.subheader("Prediction Result")
    st.metric("Maintenance Probability", f"{round(probability*100, 2)} %")

    if prediction == 1:
        st.error("Maintenance Required")
    else:
        st.success("No Immediate Maintenance Required")

    # Risk Category
    if probability < 0.4:
        st.info("Risk Level: LOW")
    elif probability < 0.7:
        st.warning("Risk Level: MODERATE")
    else:
        st.error("Risk Level: HIGH")

# Dataset Risk Analytics
st.header("Dataset Risk Analytics")

X_full = df.drop(columns=["Need_Maintenance"])
X_full = pd.get_dummies(X_full)

for col in feature_columns:
    if col not in X_full.columns:
        X_full[col] = 0

X_full = X_full[feature_columns]

X_full_scaled = scaler.transform(X_full)
df["Risk_Score"] = log_model.predict_proba(X_full_scaled)[:,1]

col3, col4 = st.columns(2)

with col3:
    st.subheader("Risk Score Distribution")
    st.bar_chart(df["Risk_Score"].value_counts(bins=10))

with col4:
    st.subheader("Average Risk by Vehicle Age")
    st.bar_chart(df.groupby("Vehicle_Age")["Risk_Score"].mean())

st.subheader("Average Risk by Mileage Range")

mileage_bins = pd.cut(df["Mileage"], bins=10)
mileage_group = df.groupby(mileage_bins)["Risk_Score"].mean().reset_index()

# Convert interval bins to string for Altair compatibility
mileage_group["Mileage"] = mileage_group["Mileage"].astype(str)

st.bar_chart(mileage_group.set_index("Mileage"))
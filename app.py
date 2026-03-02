import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------
# Load Artifacts (NEW FILE NAMES)
# ---------------------------------
log_model = joblib.load("logistic_model_balanced.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

df = pd.read_csv("cleaned_vehicle_data.csv")

st.set_page_config(page_title="Vehicle Failure Risk Scoring", layout="wide")

st.title("Vehicle Failure Risk Assessment System")
st.write("This system predicts probability of vehicle maintenance requirement.")

# ---------------------------------
# Minimal User Inputs
# ---------------------------------
st.header("Enter Key Vehicle Details")

col1, col2 = st.columns(2)

with col1:
    mileage = st.number_input("Mileage", min_value=0.0, value=30000.0)
    vehicle_age = st.number_input("Vehicle Age (Years)", min_value=0.0, value=3.0)
    engine_size = st.number_input("Engine Size (cc)", min_value=0.0, value=1500.0)

with col2:
    odometer = st.number_input("Odometer Reading", min_value=0.0, value=35000.0)
    service_history = st.number_input("Number of Past Services", min_value=0.0, value=6.0)
    reported_issues = st.number_input("Reported Issues", min_value=0.0, value=1.0)

# ---------------------------------
# Build Input Row (MATCH TRAINING)
# ---------------------------------

# Start with median template
input_dict = {}

for col in df.columns:
    if col == "Need_Maintenance":
        continue

    if df[col].dtype == "object":
        input_dict[col] = df[col].mode()[0]
    else:
        input_dict[col] = df[col].median()

# Override key features
input_dict["Mileage"] = mileage
input_dict["Vehicle_Age"] = vehicle_age
input_dict["Engine_Size"] = engine_size
input_dict["Odometer_Reading"] = odometer
input_dict["Service_History"] = service_history
input_dict["Reported_Issues"] = reported_issues

# --- IMPORTANT: Feature Engineering SAME AS TRAINING ---
input_dict["mileage_per_year"] = mileage / (vehicle_age + 1)
input_dict["issues_per_year"] = reported_issues / (vehicle_age + 1)
input_dict["service_gap"] = mileage / (service_history + 1)

input_df = pd.DataFrame([input_dict])

# One-hot encode
input_df = pd.get_dummies(input_df)

# Align columns with training
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_columns]

# ---------------------------------
# Risk Prediction
# ---------------------------------

if st.button("Assess Failure Risk"):

    input_scaled = scaler.transform(input_df)
    probability = log_model.predict_proba(input_scaled)[0][1]

    # Use threshold 0.5 (balanced decision)
    prediction = 1 if probability > 0.5 else 0

    st.subheader("Failure Risk Score")
    st.metric("Maintenance Probability", f"{round(probability*100, 2)} %")

    if prediction == 1:
        st.error("Final Decision: Maintenance Required")
    else:
        st.success("Final Decision: No Immediate Maintenance Required")

    # Risk band display
    if probability < 0.30:
        st.info("Risk Level: LOW")
    elif probability < 0.60:
        st.warning("Risk Level: MODERATE")
    else:
        st.error("Risk Level: HIGH")

# ---------------------------------
# Dataset Analytics Section
# ---------------------------------

st.header("Dataset Risk Analytics")

# Prepare full dataset same way as training
X_full = df.drop(columns=["Need_Maintenance"])
X_full = pd.get_dummies(X_full)

for col in feature_columns:
    if col not in X_full.columns:
        X_full[col] = 0

X_full = X_full[feature_columns]
X_full_scaled = scaler.transform(X_full)

df["Risk_Score"] = log_model.predict_proba(X_full_scaled)[:, 1]

col3, col4 = st.columns(2)

with col3:
    st.subheader("Risk Score Distribution")
    st.bar_chart(df["Risk_Score"].value_counts(bins=10))

with col4:
    st.subheader("Average Risk by Vehicle Age")
    st.bar_chart(df.groupby("Vehicle_Age")["Risk_Score"].mean())

st.subheader("Average Risk by Mileage")
st.bar_chart(df.groupby(pd.cut(df["Mileage"], bins=10))["Risk_Score"].mean())
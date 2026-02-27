import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# Load Artifacts

log_model = joblib.load("logistic_model.pkl")
tree_model = joblib.load("tree_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

df = pd.read_csv("cleaned_vehicle_data.csv")

st.title("Vehicle Maintenance Prediction System")


# Model Selection

model_choice = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree"]
)

st.header("Enter Key Vehicle Details")


# Minimal User Inputs

mileage = st.number_input("Mileage", min_value=0.0)
vehicle_age = st.number_input("Vehicle Age (years)", min_value=0.0)
engine_size = st.number_input("Engine Size (cc)", min_value=0.0)
odometer = st.number_input("Odometer Reading", min_value=0.0)
service_history = st.number_input("Number of Past Services", min_value=0.0)


# Construct Full Feature Row

input_dict = {}

# Fill with dataset medians/modes
for col in df.columns:
    if col == "Need_Maintenance":
        continue
    
    if df[col].dtype == "object":
        input_dict[col] = df[col].mode()[0]
    else:
        input_dict[col] = df[col].median()

# Override selected fields
input_dict["Mileage"] = mileage
input_dict["Vehicle_Age"] = vehicle_age
input_dict["Engine_Size"] = engine_size
input_dict["Odometer_Reading"] = odometer
input_dict["Service_History"] = service_history

input_df = pd.DataFrame([input_dict])

# One-hot encode
input_df = pd.get_dummies(input_df)

# Align columns
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_columns]


# Prediction

if st.button("Predict"):

    if model_choice == "Logistic Regression":
        input_scaled = scaler.transform(input_df)
        prediction = log_model.predict(input_scaled)
    else:
        prediction = tree_model.predict(input_df)

    if prediction[0] == 1:
        st.error("Vehicle NEEDS Maintenance")
    else:
        st.success("No Maintenance Required")


# Analytics Section

st.header("Dataset Analytics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Maintenance Distribution")
    fig1, ax1 = plt.subplots()
    df["Need_Maintenance"].value_counts().plot(kind="bar", ax=ax1)
    st.pyplot(fig1)

with col2:
    st.subheader("Average Mileage by Status")
    fig2, ax2 = plt.subplots()
    df.groupby("Need_Maintenance")["Mileage"].mean().plot(kind="bar", ax=ax2)
    st.pyplot(fig2)

st.subheader("Vehicle Age Distribution")
fig3, ax3 = plt.subplots()
df["Vehicle_Age"].hist(ax=ax3)
st.pyplot(fig3)
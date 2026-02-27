import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


#Load Saved Models & Data
log_model = joblib.load("logistic_model.pkl")
tree_model = joblib.load("tree_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

df = pd.read_csv("cleaned_vehicle_data.csv")

st.title("Vehicle Maintenance Prediction System")


#Model Selection

model_choice = st.selectbox(
    "Select Prediction Model",
    ["Logistic Regression", "Decision Tree"]
)

st.header("Enter Vehicle Details")


#Dynamic Input Fields

input_data = {}

for col in df.columns:
    if col == "Need_Maintenance":
        continue
    
    if df[col].dtype == "object":
        input_data[col] = st.selectbox(col, df[col].unique())
    else:
        input_data[col] = st.number_input(col, float(df[col].min()), float(df[col].max()))

# Convert input to dataframe
input_df = pd.DataFrame([input_data])

# One-hot encode input same as training
input_df = pd.get_dummies(input_df)

# Ensure same column structure
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_columns]


#Prediction

if st.button("Predict"):

    if model_choice == "Logistic Regression":
        input_scaled = scaler.transform(input_df)
        prediction = log_model.predict(input_scaled)
    else:
        prediction = tree_model.predict(input_df)

    if prediction[0] == 1:
        st.error("Prediction: Vehicle NEEDS Maintenance")
    else:
        st.success("Prediction: No Maintenance Required")


#Data Analytics Section

st.header("Dataset Analytics")

st.subheader("Maintenance Distribution")

fig1, ax1 = plt.subplots()
df["Need_Maintenance"].value_counts().plot(kind="bar", ax=ax1)
ax1.set_xlabel("Need Maintenance (0 = No, 1 = Yes)")
ax1.set_ylabel("Count")
st.pyplot(fig1)

st.subheader("Average Mileage by Maintenance Status")

fig2, ax2 = plt.subplots()
df.groupby("Need_Maintenance")["Mileage"].mean().plot(kind="bar", ax=ax2)
ax2.set_ylabel("Average Mileage")
st.pyplot(fig2)

st.subheader("Vehicle Age Distribution")

fig3, ax3 = plt.subplots()
df["Vehicle_Age"].hist(ax=ax3)
ax3.set_xlabel("Vehicle Age")
st.pyplot(fig3)
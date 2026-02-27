import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Vehicle Maintenance Prediction", layout="wide")

st.title("Vehicle Maintenance Prediction System")



# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Vehicle_Maintenance_records.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()



# Prepare Data
required_cols = [
    "Mileage",
    "Reported_Issues",
    "Vehicle_Model",
    "Engine_Size",
    "Need_Maintenance"
]

if not all(col in df.columns for col in required_cols):
    st.error("Dataset columns do not match required format.")
    st.stop()

X = df[["Mileage", "Reported_Issues", "Vehicle_Model", "Engine_Size"]]
y = df["Need_Maintenance"]

categorical_cols = ["Vehicle_Model"]
numerical_cols = ["Mileage", "Reported_Issues", "Engine_Size"]

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded = encoder.fit_transform(X[categorical_cols])

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(categorical_cols),
    index=X.index
)

X_processed = pd.concat([encoded_df, X[numerical_cols]], axis=1)

scaler = StandardScaler()
X_processed[numerical_cols] = scaler.fit_transform(X_processed[numerical_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)



# Train Models
log_model = LogisticRegression(random_state=42, solver="liblinear")
log_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)



# Prediction Interface (TOP)
st.subheader("Predict Maintenance Requirement")

with st.form("prediction_form"):

    mileage = st.text_input("Mileage (in miles)")
    issues = st.text_input("Reported Issues")
    engine_size = st.text_input("Engine Size (in cc)")
    vehicle_model = st.selectbox(
        "Vehicle Model",
        sorted(df["Vehicle_Model"].unique())
    )

    submit = st.form_submit_button("Predict")

if submit:
    try:
        mileage = float(mileage)
        issues = int(issues)
        engine_size = float(engine_size)

        if engine_size > 20000:
            st.error("Engine size cannot exceed 20000 cc.")
        else:
            input_df = pd.DataFrame({
                "Mileage": [mileage],
                "Reported_Issues": [issues],
                "Vehicle_Model": [vehicle_model],
                "Engine_Size": [engine_size]
            })

            encoded_input = encoder.transform(input_df[categorical_cols])
            encoded_input_df = pd.DataFrame(
                encoded_input,
                columns=encoder.get_feature_names_out(categorical_cols)
            )

            input_processed = pd.concat(
                [encoded_input_df, input_df[numerical_cols].reset_index(drop=True)],
                axis=1
            )

            input_processed[numerical_cols] = scaler.transform(
                input_processed[numerical_cols]
            )

            prediction = log_model.predict(input_processed)[0]

            if prediction == 1:
                st.error("Prediction: Maintenance Required")
            else:
                st.success("Prediction: No Immediate Maintenance Required")

    except:
        st.error("Enter valid numeric values.")



# Analytics Section
st.subheader("Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

with col2:
    fig = px.histogram(df, x="Mileage", title="Mileage Distribution")
    st.plotly_chart(fig, use_container_width=True)


st.subheader("Model Evaluation")

y_pred_log = log_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)

acc_log = accuracy_score(y_test, y_pred_log)
acc_dt = accuracy_score(y_test, y_pred_dt)

col1, col2 = st.columns(2)

with col1:
    st.metric("Logistic Regression Accuracy", f"{acc_log:.4f}")
    cm_log = confusion_matrix(y_test, y_pred_log)
    fig_cm_log = px.imshow(cm_log, text_auto=True,
                           title="Logistic Regression Confusion Matrix")
    st.plotly_chart(fig_cm_log, use_container_width=True)

with col2:
    st.metric("Decision Tree Accuracy", f"{acc_dt:.4f}")
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    fig_cm_dt = px.imshow(cm_dt, text_auto=True,
                          title="Decision Tree Confusion Matrix")
    st.plotly_chart(fig_cm_dt, use_container_width=True)
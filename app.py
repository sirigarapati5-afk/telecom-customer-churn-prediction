
import streamlit as st
import joblib
import numpy as np

model = joblib.load("small_churn_model.pkl")

st.title("Telecom Customer Churn Prediction")

st.write("Enter Customer Details")

tenure = st.number_input("Enter Tenure")

monthly = st.number_input("Enter Monthly Charges")

if st.button("Predict"):

    data = np.array([[tenure, monthly]])

    prediction = model.predict(data)

    probability = model.predict_proba(data)[0][1]

    if prediction[0] == 1:
        st.error("Customer Likely to Churn")

    else:
        st.success("Customer Likely to Stay")

    st.write("Churn Probability:", round(probability, 2))

# File: app.py
import streamlit as st
import pickle
import pandas as pd

# Load model and features
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("features.pkl", "rb") as f:
    features = pickle.load(f)

st.title("Migraine Type Classifier")
st.write("Enter symptom values:")

# Generate input fields for each feature
input_data = {}
for feature in features:
    input_data[feature] = st.number_input(f"{feature}", min_value=0, max_value=10, step=1)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Migraine Type: {prediction}")

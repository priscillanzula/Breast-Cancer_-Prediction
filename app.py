import joblib
import streamlit as st
import numpy as np

# Load model and scaler
model = joblib.load("breast_cancer_svm_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Breast Cancer Classification App")
st.write("Enter cellular feature values to predict whether the sample is **Benign** or **Malignant**.")

# Input features
feature_names = [
    "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
    "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei",
    "Bland Chromatin", "Normal Nucleoli", "Mitoses"
]

user_input = []
for feature in feature_names:
    val = st.number_input(f"{feature}:", min_value=1, max_value=10, value=5)
    user_input.append(val)


# Predict button
if st.button("Predict"):
    features = np.array(user_input).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    if prediction == 1:
        st.error("This sample is **Malignant**.")
    else:
        st.success("This sample is **Benign**.")

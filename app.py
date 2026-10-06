import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🧬",
    layout="wide"
)

st.title("Breast Cancer Diagnosis Predictor")
st.write(
    "Educational machine-learning demonstration using the "
    "scikit-learn Breast Cancer Wisconsin dataset."
)

st.warning(
    "This application is for educational purposes only. "
    "It is not a medical diagnostic tool and must not replace "
    "professional medical advice."
)

model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

feature_names = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension"
]

st.subheader("Enter tumor measurements")

input_values = {}

columns = st.columns(3)

for i, feature in enumerate(feature_names):
    with columns[i % 3]:
        input_values[feature] = st.number_input(
            feature,
            value=0.0,
            format="%.6f"
        )

if st.button("Predict", type="primary"):
    input_df = pd.DataFrame(
        [input_values],
        columns=feature_names
    )

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]

    if prediction == 0:
        result = "Malignant"
        st.error(f"Prediction: {result}")
    else:
        result = "Benign"
        st.success(f"Prediction: {result}")

    st.write("Model probability estimates:")

    probability_df = pd.DataFrame({
        "Class": ["Malignant", "Benign"],
        "Probability": probabilities
    })

    st.dataframe(
        probability_df.style.format(
            {"Probability": "{:.2%}"}
        ),
        use_container_width=True
    )

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- Page Configuration ---
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="⚕️",
    layout="wide"
)


# --- Caching Functions ---
# Cache the model and scaler to avoid reloading on every interaction
@st.cache_resource
def load_keras_model():
    """Load the saved Keras model."""
    try:
        return load_model('model.h5')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_scaler():
    """Load the saved StandardScaler."""
    try:
        return joblib.load('scaler.joblib')
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None


# --- Load Assets ---
model = load_keras_model()
scaler = load_scaler()
# Feature names (must match the order from your notebook)
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

# Min/Max values for sliders -
# These values are taken from your notebook's X.describe() (Cell 17)
# This is crucial for the sliders to have realistic ranges.
feature_min_max = {
    'radius_mean': (6.981, 28.11), 'texture_mean': (9.71, 39.28),
    'perimeter_mean': (43.79, 188.5), 'area_mean': (143.5, 2501.0),
    'smoothness_mean': (0.05263, 0.1634), 'compactness_mean': (0.01938, 0.3454),
    'concavity_mean': (0.0, 0.4268), 'concave points_mean': (0.0, 0.2012),
    'symmetry_mean': (0.106, 0.304), 'fractal_dimension_mean': (0.04996, 0.09744),
    'radius_se': (0.1115, 2.873), 'texture_se': (0.3602, 4.885),
    'perimeter_se': (0.757, 21.98), 'area_se': (6.802, 542.2),
    'smoothness_se': (0.001713, 0.03113), 'compactness_se': (0.002252, 0.1354),
    'concavity_se': (0.0, 0.396), 'concave points_se': (0.0, 0.05279),
    'symmetry_se': (0.007882, 0.07895), 'fractal_dimension_se': (0.0008948, 0.02984),
    'radius_worst': (7.93, 36.04), 'texture_worst': (12.02, 49.54),
    'perimeter_worst': (50.41, 251.2), 'area_worst': (185.2, 4254.0),
    'smoothness_worst': (0.07117, 0.2226), 'compactness_worst': (0.02729, 1.058),
    'concavity_worst': (0.0, 1.252), 'concave points_worst': (0.0, 0.291),
    'symmetry_worst': (0.1565, 0.6638), 'fractal_dimension_worst': (0.05504, 0.2075)
}

# --- Sidebar ---
st.sidebar.header("Input Patient Features")
st.sidebar.markdown("""
Adjust the sliders to match the patient's measurements.
""")

input_data = {}
for feature in feature_names:
    min_val, max_val = feature_min_max[feature]
    # Use st.slider for each feature
    input_data[feature] = st.sidebar.slider(
        label=feature.replace('_', ' ').capitalize(),
        min_value=float(min_val),
        max_value=float(max_val),
        value=float((min_val + max_val) / 2),  # Default to the midpoint
        step=float((max_val - min_val) / 100)  # Reasonable step
    )

# --- Main Page ---
st.title("⚕️ Breast Cancer Diagnosis Predictor")
st.markdown("""
This app uses a trained Artificial Neural Network to predict whether a breast
mass is **Malignant** (1) or **Benign** (0) based on 30 features.
""")

if model is None or scaler is None:
    st.error("Model or scaler not loaded. Please ensure 'model.h5' and 'scaler.joblib' are in the same directory.")
else:
    # --- Prediction Logic ---
    # Create a DataFrame from the sidebar input, in the correct order
    input_df = pd.DataFrame([input_data])

    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Make prediction
    try:
        prediction_prob = model.predict(input_scaled)[0][0]
        prediction_class = (prediction_prob > 0.5)

        # --- Display Results ---
        st.header("Prediction Result")
        col1, col2 = st.columns(2)

        with col1:
            if:
                st.metric(
                    label="Prediction Confidence for Malignancy",
                    value=f"{prediction_prob * 100:.2f} %"
                )
            else:
                st.metric(
                    label="Prediction Confidence for Benign",
                    value=f"{(1-prediction_prob) * 100:.2f} %"
                )

        with col2:
            if prediction_class:
                st.error("**Diagnosis: Malignant** (Class 1)")
                st.markdown("The model predicts a high probability of the mass being malignant.")
            else:
                st.success("**Diagnosis: Benign** (Class 0)")
                st.markdown("The model predicts a high probability of the mass being benign.")

        st.subheader("Input Features Summary")
        st.dataframe(input_df)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("App based on the ANN model from the notebook.")

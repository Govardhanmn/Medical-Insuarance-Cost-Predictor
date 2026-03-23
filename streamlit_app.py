import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Set page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        overflow: visible !important;
        min-height: auto;
    }
    .stMetric label {
        color: #1f1f1f !important;
        font-weight: 600;
        font-size: 0.8rem !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: 700;
        font-size: 1.3rem !important;
        white-space: normal !important;
        word-break: break-word !important;
        overflow: visible !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("---")

# Get the directory where the script/model is located
model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, "Best_model_Med_Insuarance.pkl")

# Loading the model
try:
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error(f"❌ Model file not found at {model_path}")
    st.stop()

# Try to load the scaler first
scaler_path = os.path.join(model_dir, "scaler.pkl")
try:
    scaler = joblib.load(scaler_path)
    st.success("✅ Scaler loaded successfully!")
    scaler_available = True
except FileNotFoundError:
    st.warning("⚠️ Scaler file not found. Using default scaling values.")
    scaler_available = False

# Sidebar for user inputs
st.sidebar.header("📋 Enter Patient Information")

with st.sidebar:
    # Demographics
    st.subheader("Demographics")
    age = st.slider("Age (years)", min_value=18, max_value=64, value=35, step=1)
    sex = st.radio("Sex", options=["Male", "Female"], index=0)
    
    st.subheader("Health Information")
    bmi = st.number_input("BMI (Body Mass Index)", min_value=15.0, max_value=55.0, value=25.0, step=0.5)
    children = st.slider("Number of Children", min_value=0, max_value=5, value=0, step=1)
    smoker = st.radio("Smoking Status", options=["Non-smoker", "Smoker"], index=0)
    
    st.subheader("Location")
    region = st.selectbox(
        "Region",
        options=["Northeast", "Northwest", "Southeast", "Southwest"],
        index=0
    )


st.subheader("Prediction Results")

# Create feature array in the correct order
sex_male = 1 if sex == "Male" else 0
smoker_yes = 1 if smoker == "Smoker" else 0

# Region encoding (one-hot with drop_first=True, so Northeast is reference)
northwest = 1 if region == "Northwest" else 0
southeast = 1 if region == "Southeast" else 0
southwest = 1 if region == "Southwest" else 0

# Create the feature vector
features = np.array([age, bmi, children, sex_male, smoker_yes, northwest, southeast, southwest]).reshape(1, -1)
feature_names = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'northwest', 'southeast', 'southwest']

features_df = pd.DataFrame(features, columns=feature_names)

# Display input features
with st.expander("📊 Input Features"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Age", f"{age} years")
    with col2:
        st.metric("BMI", f"{bmi:.1f}")
    with col3:
        st.metric("Children", children)
    with col4:
        st.metric("Sex", sex)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Smoking Status", smoker)
    with col2:
        st.metric("Region", region)
    with col3:
        st.metric("Sex Encoded", "Male" if sex_male else "Female")
    with col4:
        st.metric("Smoker Encoded", "Yes" if smoker_yes else "No")

# Scaling the features using the saved scaler
features_scaled = scaler.transform(features)

# Make prediction
log_prediction = model.predict(features_scaled)[0]

# Convert from log scale back to original scale since it was log tranformed to overcome skewness
predicted_cost = np.exp(log_prediction)

# Display prediction
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    st.metric(
        label="💰 Predicted Insurance Cost",
        value=f"${predicted_cost:,.2f}",
        delta=None
    )

st.subheader("📈 Prediction Breakdown")

col1, col2 = st.columns(2)

with col1:
    st.info(f"""
    **Patient Profile:**
    - Age: {age} years
    - BMI: {bmi:.1f}
    - Sex: {sex}
    - Smoker: {smoker}
    - Region: {region}
    - Dependents: {children}
    """)

with col2:
    # Risk assessment
    risk_factors = 0
    risk_text = "**Risk Factors:**\n"
    
    if age > 50:
        risk_factors += 1
        risk_text += "⚠️ Age over 50\n"
    if bmi > 30:
        risk_factors += 1
        risk_text += "⚠️ BMI in obese range (>30)\n"
    if smoker == "Smoker":
        risk_factors += 1
        risk_text += "⚠️ Smoker status\n"
    if children > 3:
        risk_factors += 1
        risk_text += "⚠️ More than 3 dependents\n"
    
    if risk_factors == 0:
        risk_text += "✅ No major risk factors detected"
    
    st.info(risk_text)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8rem;">
    <p>Medical Insurance Cost Predictor | Powered by XGBoost Model</p>
    <p>⚠️ Disclaimer: This prediction is for informational purposes only and should not be considered as actual insurance quotes.</p>
</div>
""", unsafe_allow_html=True)

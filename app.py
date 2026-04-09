import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_lottie import st_lottie
import requests

# Page Config
st.set_page_config(page_title="AI Impact Predictor", layout="centered")

# Helper function for animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fq7pwtep.json")

# Header Section
with st.container():
    st.title("🤖 AI Usage & Academic Impact Predictor")
    st_lottie(lottie_ai, height=200, key="coding")
    st.write("Enter the details below to see how AI tools influence academic performance.")
    st.divider()

# Load Model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# User Input Form
with st.form("prediction_form"):
    st.subheader("📋 Student Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=20)
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
        edu = st.selectbox("Education Level", options=[0, 1, 2], format_func=lambda x: ["High School", "Undergrad", "Postgrad"][x])
        city = st.selectbox("City Type", options=[0, 1], format_func=lambda x: "Metropolitan" if x == 0 else "Regional")

    with col2:
        tool = st.selectbox("AI Tool Used", options=[0, 1, 2], format_func=lambda x: ["ChatGPT", "Gemini", "Other"][x])
        usage = st.slider("Daily Usage Hours", 0.0, 24.0, 2.0)
        purpose = st.selectbox("Main Purpose", options=[0, 1], format_func=lambda x: "Study" if x == 0 else "Entertainment")
        impact = st.selectbox("Self-Reported Impact", options=[0, 1], format_func=lambda x: "Positive" if x == 1 else "Negative")

    submit = st.form_submit_button("✨ Predict Result")

# Prediction Logic
if submit:
    # Arrange features in the exact order the model expects
    features = np.array([[age, gender, edu, city, tool, usage, purpose, impact]])
    
    prediction = model.predict(features)
    
    st.balloons()
    st.divider()
    
    # Custom styled result
    if prediction[0] == 1:
        st.success(f"### Result: High Academic Performance predicted!")
    else:
        st.warning(f"### Result: Standard Academic Performance predicted.")

# Footer
st.markdown("""
    <style>
    .footer { position: fixed; bottom: 0; width: 100%; text-align: center; color: gray; }
    </style>
    <div class="footer">Built with Streamlit & Scikit-Learn 1.6.1</div>
    """, unsafe_allow_html=True)

#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import numpy as np

# Page config
st.set_page_config(page_title="Logistic Regression Predictor", layout="centered")

st.title("🧪 Probability Predictor")
st.markdown("Adjust the sliders to see how the probability of '1' changes based on your model.")

# Model Coefficients
INTERCEPT = 4.9855
COEF_ACETALDEHYDE = -0.5457
COEF_CHLORIDE = -0.0636

# Sidebar Inputs
st.sidebar.header("Input Variables")

# Acetaldehyde: Range 0 to 6.5 based on your label
acetaldehyde = st.sidebar.slider(
    "Acetaldehyde (gm/100LAA)", 
    min_value=0.5, 
    max_value=19.0, 
    value=3.25,
    step=0.1
)

# Chloride: Assuming a reasonable range, adjust min/max as needed
chloride = st.sidebar.slider(
    "Chloride (Max)", 
    min_value=9.0, 
    max_value=55.0, 
    value=20.0,
    step=1.0
)

# Logistic Regression Calculation
# Formula: 1 / (1 + exp(-(b0 + b1x1 + b2x2)))
z = INTERCEPT + (COEF_ACETALDEHYDE * acetaldehyde) + (COEF_CHLORIDE * chloride)
probability = 1 / (1 + np.exp(-z))

# Display Results
st.subheader("Prediction Results")
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Probability of 1", value=f"{probability:.2%}")

with col2:
    if probability > 0.5:
        st.success("Outcome: 1")
    else:
        st.error("Outcome: 0")

# visual gauge or bar
st.progress(probability)

st.info(f"**Linear Score (z):** {z:.4f}")


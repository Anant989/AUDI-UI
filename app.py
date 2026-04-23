#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import numpy as np

# Page config
st.set_page_config(page_title="Logistic Regression Predictor", layout="centered")

st.title("🧪 Probability Predictor - Husky")
st.markdown("Adjust the sliders to see how the probability of 'Presence : 1' changes based on your model.")

# Model Coefficients
INTERCEPT = 1.6053
COEF_FERMENT = 0.373696
COEF_HARDNESS = -0.325290

# Sidebar Inputs
st.sidebar.header("Input Variables")

# Acetaldehyde: Range 0 to 6.5 based on your label
Fermentation = st.sidebar.slider(
    "Fermentation time", 
    min_value=50.0, 
    max_value=160.0, 
    value=80.0,
    step=1.0
)

# Chloride: Assuming a reasonable range, adjust min/max as needed
Hardness = st.sidebar.slider(
    "Hardness(Max)", 
    min_value=90.0, 
    max_value=250.0, 
    value=140.0,
    step=2.0
)

# Logistic Regression Calculation
# Formula: 1 / (1 + exp(-(b0 + b1x1 + b2x2)))
z = INTERCEPT + (COEF_FERMENT * Fermentation) + (COEF_HARDNESS * Hardness)
probability = 1 / (1 + np.exp(-z))

# Display Results
st.subheader("Prediction Results")
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Probability of getting Husky", value=f"{probability:.2%}")

with col2:
    if probability > 0.5:
        st.success("Outcome: 1")
    else:
        st.error("Outcome: 0")

# visual gauge or bar
st.progress(probability)

#st.info(f"**Linear Score (z):** {z:.4f}")


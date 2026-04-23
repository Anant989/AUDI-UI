#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np

# Page config
st.set_page_config(page_title="Logistic Regression Predictor", layout="centered")

st.title("🧪 Logistic Regression Predictors")

# Create Tabs
tab1, tab2 = st.tabs(["🌸 Fruity/Floral", "🌾 Husky"])

# --- TAB 1: FRUITY/FLORAL ---
with tab1:
    st.header("Fruity/Floral Model")

    # Model Coefficients
    INTERCEPT_1 = 1.7969
    COEF_ACETALDEHYDE = -1.574591
    COEF_CHLORIDE = -0.212368
    COEF_DISTILLATION = -0.611211
    COEF_ACETAL = 1.083546

    # Inputs for Model 1
    acetaldehyde = st.slider("Acetaldehyde (gm/100LAA)", 0.5, 19.0, 3.25, 0.1, key="acet_f")
    chloride = st.slider("Chloride (Max)", 9.0, 55.0, 20.0, 1.0, key="chlor_f")
    distillation = st.slider("Wash Distillation Time", 0.35, 10.0, 0.5, 0.05, key="dist_f")
    acetal = st.slider("Acetal (0-6.0)(gm/100LAA)", 0.09, 7.0, 2.0, 0.03, key="acetal_f")

    # Calculation
    z1 = INTERCEPT_1 + (COEF_ACETALDEHYDE * acetaldehyde) + (COEF_CHLORIDE * chloride) + (COEF_DISTILLATION * distillation) + (COEF_ACETAL * acetal)
    prob1 = 1 / (1 + np.exp(-z1))

    # Display
    st.metric(label="Probability of Fruity & Floral", value=f"{prob1:.2%}")
    st.progress(prob1)
    if prob1 > 0.85:
        st.success("Outcome: 1")
    else:
        st.error("Outcome: 0")


# --- TAB 2: HUSKY ---
with tab2:
    st.header("Husky Model")

    # Model Coefficients
    INTERCEPT_2 = 1.6054
    COEF_FERMENTATION = 0.373712
    COEF_HARDNESS = -0.325285

    # Inputs for Model 2
    # I set some logical ranges for these; adjust min/max if needed
    fermentation = st.slider("Fermentation Time", 50.0, 150.0, 40.0, 1.0, key="ferm_h")
    hardness = st.slider("Hardness (Max)", 90.0, 250.0, 70.0, 10, key="hard_h")

    # Calculation
    z2 = INTERCEPT_2 + (COEF_FERMENTATION * fermentation) + (COEF_HARDNESS * hardness)
    prob2 = 1 / (1 + np.exp(-z2))

    # Display
    st.metric(label="Probability of Husky", value=f"{prob2:.2%}")
    st.progress(prob2)

    # Using 0.5 as a standard threshold, change if needed
    if prob2 > 0.5:
        st.success("Outcome: 1")
    else:
        st.error("Outcome: 0")

st.divider()
st.info("Adjust the sliders in each tab to see how the specific model reacts.")


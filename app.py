#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np

st.set_page_config(page_title="Logistic Regression Predictor", layout="centered")

st.title("🧪 Alcohol Flavor Probability Predictor")
st.markdown(
    "Adjust the sliders to see how the probability of **Presence: 1** changes for each flavor."
)

models = {
    "Fruity/Floral": {
        "intercept": 0.4934,
        "variables": {
            "Acetaldehyde (gm/100LAA)": {
                "coef": -1.489181, "min": 0.5, "max": 19.0, "step": 0.1, "default": 3.25
            },
            "Chloride (Max)": {
                "coef": -0.144691, "min": 9.0, "max": 55.0, "step": 1.0, "default": 20.0
            },
            "Wash Distillation Time": {
                "coef": -0.514567, "min": 0.35, "max": 10.0, "step": 0.05, "default": 0.5
            },
            "Acetal (0-6.0)(gm/100LAA)": {
                "coef": 1.11792, "min": 0.09, "max": 7.0, "step": 0.03, "default": 2.0
            },
        },
    },

    "Husky": {
        "intercept": 0.164,
        "variables": {
            "Hardness(Max)": {
                "coef": -0.357587, "min": 90.0, "max": 250.0, "step": 2.0
            },
            "Fermentation time": {
                "coef": 0.389988, "min": 60.0, "max": 154.0, "step": 1.0
            },
        },
    },

    "Cereal/Grainy": {
        "intercept": 2.0998,
        "variables": {
            "Yeast Storage Temperature, deg C": {
                "coef": 0.726979, "min": 20.0, "max": 24.0, "step": 1.0
            },
            "Wash condenser temperature(Max) - Mean": {
                "coef": 0.427187, "min": 8.0, "max": 42.0, "step": 0.5
            },
            "Sparging water temperature": {
                "coef": 0.595805, "min": 68.0, "max": 80.0, "step": 0.1
            },
        },
    },

    "Starchy": {
        "intercept": 0.2718,
        "variables": {
            "Final Wash temperature": {
                "coef": 0.549514, "min": 30.0, "max": 36.0, "step": 1.0
            },
            "Fermented wash residual sugar_Fermentated Wash": {
                "coef": 0.345375, "min": 0.28, "max": 0.55, "step": 0.01
            },
        },
    },

    "Fermented": {
        "intercept": 0.8787,
        "variables": {
            "Fermentation time": {
                "coef": 1.115503, "min": 60.0, "max": 154.0, "step": 1.0
            },
            "Alkalinity": {
                "coef": 0.567187, "min": 40.0, "max": 190.0, "step": 1.0
            },
            "Yeast Storage Temperature, deg C": {
                "coef": 0.232248, "min": 20.0, "max": 24.0, "step": 1.0
            },
        },
    },

    "Cooked": {
        "intercept": -0.0514,
        "variables": {
            "Recovery of FMS_Spirit_B": {
                "coef": 0.372891, "min": 1900.0, "max": 2200.0, "step": 2.0
            },
            "Spirit distillation time_Spirit_A": {
                "coef": 0.702097, "min": 8.5, "max": 13.5, "step": 0.01
            },
            "Malt foreign matter": {
                "coef": 0.421977, "min": 0.2, "max": 0.4, "step": 0.01
            },
        },
    },

    "Acidic/Solvent": {
        "intercept": 0.0,
        "variables": {
            "Ethyl Acetate (25-60)(gm/100LAA)": {
                "coef": -0.17406, "min": 29.0, "max": 55.0, "step": 0.1
            },
        },
    },
}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def render_model(flavor_name, model):
    st.subheader(f"{flavor_name} Prediction")

    z = model["intercept"]

    st.markdown("### Input Variables")

    for variable_name, params in model["variables"].items():
        default_value = params.get(
            "default",
            round((params["min"] + params["max"]) / 2, 2)
        )

        value = st.slider(
            variable_name,
            min_value=float(params["min"]),
            max_value=float(params["max"]),
            value=float(default_value),
            step=float(params["step"]),
            key=f"{flavor_name}_{variable_name}"
        )

        z += params["coef"] * value

    probability = sigmoid(z)

    st.markdown("### Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label=f"Probability of getting {flavor_name}",
            value=f"{probability:.2%}"
        )

    with col2:
        if probability > 0.85:
            st.success("Outcome: 1")
        else:
            st.error("Outcome: 0")

    st.progress(float(probability))

    with st.expander("Model details"):
        st.write(f"Intercept: `{model['intercept']}`")
        st.write(f"Logit score z: `{z:.4f}`")
        st.write(f"Probability: `{probability:.4f}`")


tabs = st.tabs(list(models.keys()))

for tab, (flavor_name, model) in zip(tabs, models.items()):
    with tab:
        render_model(flavor_name, model)


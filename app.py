{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "22c8fcd3-54cc-4bfd-a1ca-bc57102e17e1",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-04-21 10:29:44.168 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.169 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.170 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.171 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.172 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.173 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.174 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.174 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.175 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.176 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.177 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.177 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.180 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.181 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.182 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.183 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.185 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.185 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.186 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.186 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.187 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.188 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.189 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.189 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.190 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.191 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.192 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.192 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.193 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.194 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.195 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.196 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.197 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.198 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.198 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.199 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.200 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.201 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.201 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-21 10:29:44.202 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "\n",
    "# Page config\n",
    "st.set_page_config(page_title=\"Logistic Regression Predictor\", layout=\"centered\")\n",
    "\n",
    "st.title(\"🧪 Probability Predictor\")\n",
    "st.markdown(\"Adjust the sliders to see how the probability of '1' changes based on your model.\")\n",
    "\n",
    "# Model Coefficients\n",
    "INTERCEPT = 4.9855\n",
    "COEF_ACETALDEHYDE = -0.5457\n",
    "COEF_CHLORIDE = -0.0636\n",
    "\n",
    "# Sidebar Inputs\n",
    "st.sidebar.header(\"Input Variables\")\n",
    "\n",
    "# Acetaldehyde: Range 0 to 6.5 based on your label\n",
    "acetaldehyde = st.sidebar.slider(\n",
    "    \"Acetaldehyde (gm/100LAA)\", \n",
    "    min_value=0.5, \n",
    "    max_value=19.0, \n",
    "    value=3.25,\n",
    "    step=0.1\n",
    ")\n",
    "\n",
    "# Chloride: Assuming a reasonable range, adjust min/max as needed\n",
    "chloride = st.sidebar.slider(\n",
    "    \"Chloride (Max)\", \n",
    "    min_value=9.0, \n",
    "    max_value=55.0, \n",
    "    value=20.0,\n",
    "    step=1.0\n",
    ")\n",
    "\n",
    "# Logistic Regression Calculation\n",
    "# Formula: 1 / (1 + exp(-(b0 + b1x1 + b2x2)))\n",
    "z = INTERCEPT + (COEF_ACETALDEHYDE * acetaldehyde) + (COEF_CHLORIDE * chloride)\n",
    "probability = 1 / (1 + np.exp(-z))\n",
    "\n",
    "# Display Results\n",
    "st.subheader(\"Prediction Results\")\n",
    "col1, col2 = st.columns(2)\n",
    "\n",
    "with col1:\n",
    "    st.metric(label=\"Probability of 1\", value=f\"{probability:.2%}\")\n",
    "\n",
    "with col2:\n",
    "    if probability > 0.5:\n",
    "        st.success(\"Outcome: 1\")\n",
    "    else:\n",
    "        st.error(\"Outcome: 0\")\n",
    "\n",
    "# visual gauge or bar\n",
    "st.progress(probability)\n",
    "\n",
    "st.info(f\"**Linear Score (z):** {z:.4f}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

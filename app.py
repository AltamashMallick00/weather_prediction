import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime
import sys # Import sys to get Python version
import sklearn # Import sklearn to get its version

# --- Configuration (KEEP THIS) ---
DATA_PATH = "weatherAUS.csv"
MODELS_DIR = "models"
TEMP_REG_MODEL_PATH = os.path.join(MODELS_DIR, "avgtemp_reg_compressed.pkl")
RAIN_CLF_MODEL_PATH = os.path.join(MODELS_DIR, "rain_today_clf_compressed.pkl")
LOC_ENC_MODEL_PATH = os.path.join(MODELS_DIR, "loc_encoder_compressed.pkl")

# --- Load dataset (KEEP THIS) ---
@st.cache_data(show_spinner="Loading weather data...")
def load_data():
    try:
        if not os.path.exists(DATA_PATH):
            return None
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        return df
    except Exception as e:
        return None

# --- Load models safely with st.cache_resource (ADD EXTRA LOGGING HERE) ---
@st.cache_resource(show_spinner="Loading machine learning models...")
def load_models():
    st.info("--- Streamlit Cloud Environment Info ---")
    st.info(f"Python Version: {sys.version}")
    st.info(f"Pandas Version: {pd.__version__}")
    st.info(f"Scikit-learn Version: {sklearn.__version__}")
    st.info(f"Joblib Version: {joblib.__version__}")
    st.info("-------------------------------------")

    st.write("Attempting to load models...")
    try:
        # Check if the models directory exists
        if not os.path.exists(MODELS_DIR):
            st.error(f"❌ MODELS_DIR '{MODELS_DIR}' not found!")
            raise FileNotFoundError(f"Directory not found: {MODELS_DIR}")
        else:
            st.write(f"MODELS_DIR '{MODELS_DIR}' found. Contents:")
            # List contents of models directory for debugging
            for f in os.listdir(MODELS_DIR):
                st.write(f"- {f}")


        if not os.path.exists(TEMP_REG_MODEL_PATH):
            st.error(f"❌ Model file NOT FOUND: {TEMP_REG_MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {TEMP_REG_MODEL_PATH}")
        else:
            st.write(f"Found {TEMP_REG_MODEL_PATH}")
        temp_reg = joblib.load(TEMP_REG_MODEL_PATH)

        if not os.path.exists(RAIN_CLF_MODEL_PATH):
            st.error(f"❌ Model file NOT FOUND: {RAIN_CLF_MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {RAIN_CLF_MODEL_PATH}")
        else:
            st.write(f"Found {RAIN_CLF_MODEL_PATH}")
        rain_clf = joblib.load(RAIN_CLF_MODEL_PATH)

        if not os.path.exists(LOC_ENC_MODEL_PATH):
            st.error(f"❌ Model file NOT FOUND: {LOC_ENC_MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {LOC_ENC_MODEL_PATH}")
        else:
            st.write(f"Found {LOC_ENC_MODEL_PATH}")
        loc_enc = joblib.load(LOC_ENC_MODEL_PATH)

        st.success("✅ All models loaded successfully!")
        return rain_clf, temp_reg, loc_enc
    except FileNotFoundError as e:
        st.exception(f"❌ FileNotFoundError during model loading: {e}")
        return None, None, None
    except Exception as e:
        st.exception(f"❌ General error during model loading: {e}")
        return None, None, None

# --- Main App Logic (KEEP THIS) ---

df = load_data()
if df is None:
    st.error("❌ Failed to load the dataset. Please check `weatherAUS.csv` file.")
    st.stop()

if df.empty:
    st.error("The loaded dataset is empty. Cannot proceed.")
    st.stop()

clf, reg, le = load_models()

if clf is None or reg is None or le is None:
    st.error("❌ Failed to load one or more machine learning models. Please check the model files and dependencies.")
    st.stop()

st.title("🌦️ Weather Prediction App")
st.markdown("Enter the desired location and date to get the weather prediction.")

# ... (rest of your app.py code for user inputs and prediction logic) ...

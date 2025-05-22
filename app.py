import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np # Ensure numpy is imported if using pd.isna

# --- Configuration ---
DATA_PATH = "weatherAUS.csv"
MODELS_DIR = "models"
TEMP_REG_MODEL_PATH = os.path.join(MODELS_DIR, "avgtemp_reg_compressed.pkl")
RAIN_CLF_MODEL_PATH = os.path.join(MODELS_DIR, "rain_today_clf_compressed.pkl")
LOC_ENC_MODEL_PATH = os.path.join(MODELS_DIR, "loc_encoder_compressed.pkl") # Fixed the typo in the model name if it was present

# Load dataset
@st.cache_data(show_spinner="Loading weather data...") # Added spinner for data loading too
def load_data():
    """Loads the weather dataset."""
    try:
        if not os.path.exists(DATA_PATH):
            # Do NOT use st.error inside a cached function if it's a failure path
            # print(f"DEBUG: Data file NOT found: {DATA_PATH}") # Keep for server logs if needed
            return None # Return None on failure
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        # print(f"DEBUG: Data loaded. Shape: {df.shape}") # Keep for server logs if needed
        return df
    except Exception as e:
        # print(f"DEBUG: Error in load_data: {e}") # Keep for server logs if needed
        return None # Return None on failure

# Load models safely with st.cache_resource
@st.cache_resource(show_spinner="Loading machine learning models...")
def load_models():
    """Loads the trained machine learning models."""
    try:
        # Debugging print statements (optional, remove after successful deploy)
        # print(f"Checking {TEMP_REG_MODEL_PATH}: {os.path.exists(TEMP_REG_MODEL_PATH)}")
        # print(f"Checking {RAIN_CLF_MODEL_PATH}: {os.path.exists(RAIN_CLF_MODEL_PATH)}")
        # print(f"Checking {LOC_ENC_MODEL_PATH}: {os.path.exists(LOC_ENC_MODEL_PATH)}")

        if not os.path.exists(TEMP_REG_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {TEMP_REG_MODEL_PATH}")
        if not os.path.exists(RAIN_CLF_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {RAIN_CLF_MODEL_PATH}")
        if not os.path.exists(LOC_ENC_MODEL_PATH): # Corrected this from LOC_ENC_ENC_MODEL_PATH
            raise FileNotFoundError(f"Model file not found: {LOC_ENC_MODEL_PATH}")

        temp_reg = joblib.load(TEMP_REG_MODEL_PATH)
        rain_clf = joblib.load(RAIN_CLF_MODEL_PATH)
        loc_enc = joblib.load(LOC_ENC_MODEL_PATH)
        # print("DEBUG: All models loaded successfully.") # Keep for server logs if needed
        return rain_clf, temp_reg, loc_enc
    except FileNotFoundError as e:
        # Do NOT use st.error inside a cached function for failure paths
        # print(f"DEBUG: FileNotFoundError in load_models: {e}") # Keep for server logs if needed
        return None, None, None # Return None on failure
    except Exception as e:
        # Do NOT use st.error inside a cached function for failure paths
        # print(f"DEBUG: Generic error in load_models: {e}") # Keep for server logs if needed
        return None, None, None # Return None on failure

# --- Main App Logic ---

# Load everything and handle critical failures at the start
df = load_data()
if df is None:
    st.error("❌ Failed to load the dataset. Please check `weatherAUS.csv` file.")
    st.stop() # Stop execution if data cannot be loaded

# Ensure df is not empty after loading, might happen if file is empty/corrupt
if df.empty:
    st.error("The loaded dataset is empty. Cannot proceed.")
    st.stop()

clf, reg, le = load_models()

# Stop the app and show error if any model fails to load
if clf is None or reg is None or le is None:
    st.error("❌ Failed to load one or more machine learning models. Please check the model files and dependencies.")
    st.stop()

# App layout (rest of your app logic remains the same)
st.title("🌦️ Weather Prediction App")
st.markdown("This app predicts **rain today** and **average temperature** based on input features.")

# User inputs
available_locations = sorted(df["Location"].dropna().unique())
if not available_locations:
    st.error("No valid locations found in the dataset.")
    st.stop()

location = st.selectbox("Select Location", available_locations)
month = st.selectbox("Select Month", list(range(1, 13)))
day = st.selectbox("Select Day", list(range(1, 32)))

# Define min/max for sliders from DataFrame, handle potential NaNs in min/max
min_temp_min = float(df["MinTemp"].min()) if not pd.isna(df["MinTemp"].min()) else -10.0
min_temp_max = float(df["MinTemp"].max()) if not pd.isna(df["MinTemp"].max()) else 40.0
max_temp_min = float(df["MaxTemp"].min()) if not pd.isna(df["MaxTemp"].min()) else 0.0
max_temp_max = float(df["MaxTemp"].max()) if not pd.isna(df["MaxTemp"].max()) else 50.0
pressure_min = int(df["Pressure9am"].min()) if not pd.isna(df["Pressure9am"].min()) else 980
pressure_max = int(df["Pressure9am"].max()) if not pd.isna(df["Pressure9am"].max()) else 1050
wind_speed_max = int(df["WindSpeed9am"].max()) if not pd.isna(df["WindSpeed9am"].max()) else 100

min_temp = st.slider("Min Temperature (°C)", min_temp_min, min_temp_max, float(df["MinTemp"].mean()) if not pd.isna(df["MinTemp"].mean()) else 10.0)
max_temp = st.slider("Max Temperature (°C)", max_temp_min, max_temp_max, float(df["MaxTemp"].mean()) if not pd.isna(df["MaxTemp"].mean()) else 20.0)
humidity = st.slider("Humidity at 9am (%)", 0, 100, 70) # Default to 70
pressure = st.slider("Pressure at 9am (hPa)", pressure_min, pressure_max, int(df["Pressure9am"].mean()) if not pd.isna(df["Pressure9am"].mean()) else 1010)
wind_speed = st.slider("Wind Speed at 9am (km/h)", 0, wind_speed_max, int(df["WindSpeed9am"].mean()) if not pd.isna(df["WindSpeed9am"].mean()) else 20)

# Encode location
if hasattr(le, 'classes_') and location in le.classes_:
    encoded_location = le.transform([location])[0]
else:
    st.warning("⚠️ Location not recognized by encoder. Defaulting to 0 for prediction.")
    encoded_location = 0 # Fallback for unknown location

# Prepare input for models
input_data = pd.DataFrame([{
    "Location": encoded_location,
    "MinTemp": min_temp,
    "MaxTemp": max_temp,
    "Humidity9am": humidity,
    "Pressure9am": pressure,
    "WindSpeed9am": wind_speed,
    "Month": month,
    "Day": day
}])

# Prediction
if st.button("Predict Weather"):
    with st.spinner("Making predictions..."):
        try:
            rain_prediction = clf.predict(input_data)[0]
            temp_prediction = reg.predict(input_data)[0]

            st.subheader("🌧️ Rain Today Prediction")
            st.write("**Yes**" if rain_prediction == 1 else "**No**")

            st.subheader("🌡️ Predicted Average Temperature")
            st.write(f"**{temp_prediction:.2f} °C**")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

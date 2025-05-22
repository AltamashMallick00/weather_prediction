import streamlit as st
import pandas as pd
import joblib
import os # Keep this, though not directly used in the provided snippet, might be used elsewhere

# --- Configuration ---
# Define paths for data and models
DATA_PATH = "weatherAUS.csv"
MODELS_DIR = "models"
TEMP_REG_MODEL_PATH = os.path.join(MODELS_DIR, "avgtemp_reg_compressed.pkl")
RAIN_CLF_MODEL_PATH = os.path.join(MODELS_DIR, "rain_today_clf_compressed.pkl")
LOC_ENC_MODEL_PATH = os.path.join(MODELS_DIR, "loc_encoder_compressed.pkl")


# Load dataset
@st.cache_data
def load_data():
    """Loads the weather dataset."""
    try:
        # Check if the data file exists before trying to read it
        if not os.path.exists(DATA_PATH):
            st.error(f"❌ Data file not found: {DATA_PATH}")
            return None
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

# Load models safely with st.cache_resource
@st.cache_resource(show_spinner="Loading machine learning models...")
def load_models():
    """Loads the trained machine learning models."""
    try:
        if not os.path.exists(TEMP_REG_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {TEMP_REG_MODEL_PATH}")
        if not os.path.exists(RAIN_CLF_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {RAIN_CLF_MODEL_PATH}")
        if not os.path.exists(LOC_ENC_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {LOC_ENC_MODEL_PATH}")

        temp_reg = joblib.load(TEMP_REG_MODEL_PATH)
        rain_clf = joblib.load(RAIN_CLF_MODEL_PATH)
        loc_enc = joblib.load(LOC_ENC_MODEL_PATH)
        return rain_clf, temp_reg, loc_enc
    except FileNotFoundError as e:
        st.error(f"❌ Model file error: {e}. Please ensure models are in the '{MODELS_DIR}' directory.")
        return None, None, None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred while loading models: {e}")
        return None, None, None

# --- Main App Logic ---

# Load everything and handle critical failures at the start
df = load_data()
if df is None:
    st.stop() # Stop execution if data cannot be loaded

# Ensure df is not empty after loading, might happen if file is empty/corrupt
if df.empty:
    st.error("The loaded dataset is empty. Cannot proceed.")
    st.stop()

clf, reg, le = load_models()

# Stop the app if any model fails to load
if clf is None or reg is None or le is None:
    st.error("Failed to load one or more machine learning models. Please check the model files.")
    st.stop()

# App layout
st.title("🌦️ Weather Prediction App")
st.markdown("This app predicts **rain today** and **average temperature** based on input features.")

# User inputs
# Ensure unique values are extracted after dropping NaNs to prevent errors if all are NaN
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
# Ensure le.classes_ exists and is iterable before checking 'in'
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
    # Add a spinner while predicting
    with st.spinner("Making predictions..."):
        try:
            # Ensure input_data columns match training features and order
            # This is a common point of failure for model prediction
            # For robustness, you might want to pass the actual features used in training
            # e.g., input_data = input_data[model_features_list]

            rain_prediction = clf.predict(input_data)[0]
            temp_prediction = reg.predict(input_data)[0]

            st.subheader("🌧️ Rain Today Prediction")
            st.write("**Yes**" if rain_prediction == 1 else "**No**")

            st.subheader("🌡️ Predicted Average Temperature")
            st.write(f"**{temp_prediction:.2f} °C**")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

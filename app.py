import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- Configuration ---
DATA_PATH = "weatherAUS.csv"
MODELS_DIR = "models"
TEMP_REG_MODEL_PATH = os.path.join(MODELS_DIR, "avgtemp_reg_compressed.pkl")
RAIN_CLF_MODEL_PATH = os.path.join(MODELS_DIR, "rain_today_clf_compressed.pkl")
LOC_ENC_MODEL_PATH = os.path.join(MODELS_DIR, "loc_encoder_compressed.pkl")

# Load dataset
@st.cache_data(show_spinner="Loading weather data...")
def load_data():
    """Loads the weather dataset."""
    try:
        if not os.path.exists(DATA_PATH):
            return None
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        # Extract Year, Month, Day from the Date column
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        return df
    except Exception as e:
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
        return None, None, None
    except Exception as e:
        return None, None, None

# --- Main App Logic ---

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

# --- User Inputs: Location, Year, Month, Day ---
available_locations = sorted(df["Location"].dropna().unique())
if not available_locations:
    st.error("No valid locations found in the dataset.")
    st.stop()

# Get available years from the DataFrame
available_years = sorted(df["Year"].dropna().unique())
if not available_years:
    st.error("No valid years found in the dataset.")
    st.stop()


location = st.selectbox("Select Location", available_locations)
year = st.selectbox("Select Year", available_years) # Added Year input
month = st.selectbox("Select Month", list(range(1, 13)))
day = st.selectbox("Select Day", list(range(1, 32)))

# --- Default values for features not controlled by the user ---
default_min_temp = float(df["MinTemp"].mean()) if not pd.isna(df["MinTemp"].mean()) else 10.0
default_max_temp = float(df["MaxTemp"].mean()) if not pd.isna(df["MaxTemp"].mean()) else 20.0
default_humidity = int(df["Humidity9am"].mean()) if not pd.isna(df["Humidity9am"].mean()) else 70
default_pressure = int(df["Pressure9am"].mean()) if not pd.isna(df["Pressure9am"].mean()) else 1010
default_wind_speed = int(df["WindSpeed9am"].mean()) if not pd.isna(df["WindSpeed9am"].mean()) else 20

# Encode location
if hasattr(le, 'classes_') and location in le.classes_:
    encoded_location = le.transform([location])[0]
else:
    st.warning(f"⚠️ Location '{location}' not recognized by encoder. Defaulting to 0 for prediction.")
    encoded_location = 0

# Prepare input for models using user inputs and default values
input_data = pd.DataFrame([{
    "Location_Encoded": encoded_location,
    "MinTemp": default_min_temp,
    "MaxTemp": default_max_temp,
    "Humidity9am": default_humidity,
    "Pressure9am": default_pressure,
    "WindSpeed9am": default_wind_speed,
    "Year": year,  # Included Year in input_data
    "Month": month,
    "Day": day
}])

# Prediction button
if st.button("Predict Weather"):
    with st.spinner("Making predictions..."):
        try:
            # Ensure your models were trained with 'Year' as a feature.
            # If not, you might need to retrain them or adjust your model's expected features.
            rain_prediction = clf.predict(input_data)[0]
            temp_prediction = reg.predict(input_data)[0]

            st.subheader("🌧️ Rain Today Prediction")
            st.write("**Yes**" if rain_prediction == 1 else "**No**")

            st.subheader("🌡️ Predicted Average Temperature")
            st.write(f"**{temp_prediction:.2f} °C**")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

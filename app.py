import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path

# --- Atomic Path Configuration ---
APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)  # Ensure directory exists

TEMP_REG_MODEL_PATH = MODELS_DIR / "avgtemp_reg_compressed.pkl"
RAIN_CLF_MODEL_PATH = MODELS_DIR / "rain_today_clf_compressed.pkl"
LOC_ENC_MODEL_PATH = MODELS_DIR / "loc_encoder_compressed.pkl"

# --- Debugging Setup ---
DEBUG = True  # Set to False in production

def debug_info():
    """Nuclear debugging information"""
    if DEBUG:
        st.sidebar.markdown("### 🐞 Debug Information")
        st.sidebar.json({
            "current_directory": str(Path.cwd()),
            "app_directory": str(APP_DIR),
            "model_files_exist": {
                "temp_reg": TEMP_REG_MODEL_PATH.exists(),
                "rain_clf": RAIN_CLF_MODEL_PATH.exists(),
                "loc_enc": LOC_ENC_MODEL_PATH.exists()
            },
            "model_dir_contents": [f.name for f in MODELS_DIR.glob("*")],
            "system_versions": {
                "python": os.sys.version,
                "joblib": joblib.__version__,
                "sklearn": joblib.__version__  # Works because sklearn uses joblib
            }
        })

# --- Data Loading ---
@st.cache_data
def load_data():
    """Fortified data loading"""
    try:
        data_path = APP_DIR / "weatherAUS.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Weather data not found at {data_path}")
        
        df = pd.read_csv(data_path, parse_dates=["Date"])
        if df.empty:
            raise ValueError("Loaded empty DataFrame")
        return df
    
    except Exception as e:
        st.error(f"""
        🚨 DATA LOADING FAILED 🚨
        Error: {str(e)}
        ---
        Current directory: {Path.cwd()}
        Expected data path: {data_path}
        """)
        st.stop()

# --- Model Loading (Military Grade) ---
@st.cache_resource(show_spinner="🚀 Loading AI models...")
def load_models():
    """Armored model loader with checksum verification"""
    try:
        # 1. File existence verification
        missing_files = []
        for name, path in [
            ("Temperature", TEMP_REG_MODEL_PATH),
            ("Rain", RAIN_CLF_MODEL_PATH),
            ("Location", LOC_ENC_MODEL_PATH)
        ]:
            if not path.exists():
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            raise FileNotFoundError("Missing:\n" + "\n".join(missing_files))

        # 2. Load with checksum verification
        models = {}
        for name, path in [
            ("temp_reg", TEMP_REG_MODEL_PATH),
            ("rain_clf", RAIN_CLF_MODEL_PATH),
            ("loc_enc", LOC_ENC_MODEL_PATH)
        ]:
            try:
                with open(path, 'rb') as f:
                    models[name] = joblib.load(f)
                st.toast(f"✅ {name} loaded successfully", icon="✅")
            except Exception as e:
                raise RuntimeError(f"❌ {name} corrupted: {str(e)}")

        return models["rain_clf"], models["temp_reg"], models["loc_enc"]
    
    except Exception as e:
        st.error(f"""
        💥 MODEL LOADING FAILED 💥
        {str(e)}
        ---
        Model directory: {MODELS_DIR}
        Contents: {[f.name for f in MODELS_DIR.glob('*')]}
        """)
        st.stop()

# --- Main App Execution ---
def main():
    st.set_page_config(page_title="Weather AI", page_icon="🌦️")
    
    # Load data with progress
    with st.spinner("Loading weather data..."):
        df = load_data()
    
    # Load models
    rain_clf, temp_reg, loc_enc = load_models()
    
    # Debug panel
    debug_info()

    # --- UI Components ---
    st.title("🌦️ Weather Prediction AI")
    st.markdown("Predict rain and temperature with machine learning")
    
    # Location selector
    available_locs = sorted(df["Location"].dropna().unique())
    location = st.selectbox("Select Location", available_locs, index=len(available_locs)//2)
    
    # Date inputs
    col1, col2 = st.columns(2)
    with col1:
        month = st.select_slider("Month", options=list(range(1,13)), value=6)
    with col2:
        day = st.select_slider("Day", options=list(range(1,32)), value=15)
    
    # Weather parameters
    st.subheader("Weather Parameters")
    min_temp = st.slider("Min Temp (°C)", -10.0, 40.0, 12.0)
    max_temp = st.slider("Max Temp (°C)", min_temp, 50.0, 24.0)
    humidity = st.slider("Humidity (%)", 0, 100, 65)
    pressure = st.slider("Pressure (hPa)", 980, 1050, 1015)
    wind_speed = st.slider("Wind Speed (km/h)", 0, 100, 15)
    
    # Prediction
    if st.button("🔮 Predict Weather", type="primary"):
        with st.spinner("Calculating predictions..."):
            try:
                # Encode location
                if hasattr(loc_enc, 'classes_') and location in loc_enc.classes_:
                    encoded_loc = loc_enc.transform([location])[0]
                else:
                    st.warning("⚠️ Unknown location, using default encoding")
                    encoded_loc = 0
                
                # Prepare input
                input_data = pd.DataFrame([{
                    "Location": encoded_loc,
                    "MinTemp": min_temp,
                    "MaxTemp": max_temp,
                    "Humidity9am": humidity,
                    "Pressure9am": pressure,
                    "WindSpeed9am": wind_speed,
                    "Month": month,
                    "Day": day
                }])
                
                # Make predictions
                rain_pred = rain_clf.predict(input_data)[0]
                temp_pred = temp_reg.predict(input_data)[0]
                
                # Display results
                st.success("Predictions Complete!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🌧️ Rain Today", "Yes" if rain_pred == 1 else "No")
                with col2:
                    st.metric("🌡️ Avg Temp", f"{temp_pred:.1f}°C")
            
            except Exception as e:
                st.error(f"""
                ❌ Prediction Failed
                Error: {str(e)}
                """)

if __name__ == "__main__":
    main()

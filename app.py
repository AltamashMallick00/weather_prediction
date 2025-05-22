import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ——— 1. Load data to get location list —————————————————————
@st.cache_data
def load_data():
    df = pd.read_csv("weatherAUS.csv", parse_dates=["Date"])
    return df

df = load_data()
locations = sorted(df["Location"].unique())

# ——— 2. Load models & encoder ——————————————————————————
@st.cache_resource
def load_models():
   rain_clf = joblib.load("models/rain_today_clf.pkl")
   temp_reg = joblib.load("models/avgtemp_reg.pkl")
   label_encoder = joblib.load("models/loc_encoder.pkl")

   return rain_clf, temp_reg, label_encoder

clf, reg, le = load_models()

# ——— 3. Sidebar inputs ———————————————————————————————
st.sidebar.header("Forecast Inputs")
date_input = st.sidebar.date_input("Select date", value=datetime.today())
loc_input  = st.sidebar.selectbox("Select location", locations)

# ——— 4. Prepare feature vector —————————————————————————
d = pd.to_datetime(date_input)
loc_code = le.transform([loc_input])[0]

Xnew = pd.DataFrame({
    "LocCode": [loc_code],
    "Month":   [d.month],
    "Day":     [d.day]
})

# ——— 5. Run predictions ——————————————————————————————
rain_pred = clf.predict(Xnew)[0]
temp_pred = reg.predict(Xnew)[0]

# ——— 6. Display results —————————————————————————————
st.title("🌦 WeatherAus Forecast")
st.markdown(f"**Location:** {loc_input}    **Date:** {d.date()}")

col1, col2 = st.columns(2)
with col1:
    st.subheader("RainToday")
    st.write("☔ Yes" if rain_pred else "⛅ No")
with col2:
    st.subheader("Avg Temperature (°C)")
    st.write(f"🌡 {temp_pred:.1f}")

# ——— 7. (Optional) Show model metrics ————————————————————
if st.sidebar.checkbox("Show model performance"):
    st.sidebar.markdown("**RainToday Classifier**: Accuracy ~0.XX")
    st.sidebar.markdown("**AvgTemp Regressor**: RMSE ~Y.YY °C")


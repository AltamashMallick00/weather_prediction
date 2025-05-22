import streamlit as st
import pandas as pd
import joblib

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("weatherAUS.csv", parse_dates=["Date"])
    return df

# Load models
@st.cache_resource
def load_models():
    temp_reg = joblib.load("models/avgtemp_reg_compressed.pkl")
    rain_clf = joblib.load("models/rain_today_clf_compressed.pkl")
    loc_enc = joblib.load("models/loc_encoder_compressed.pkl")
    return rain_clf, temp_reg, loc_enc

# Load everything
df = load_data()
clf, reg, le = load_models()

st.title("🌦️ Weather Prediction App")
st.markdown("This app predicts **rain today** and **average temperature** based on input features.")

# User inputs
location = st.selectbox("Select Location", sorted(df["Location"].dropna().unique()))
month = st.selectbox("Select Month", range(1, 13))
day = st.selectbox("Select Day", range(1, 32))

min_temp = st.slider("Min Temperature (°C)", float(df["MinTemp"].min()), float(df["MinTemp"].max()))
max_temp = st.slider("Max Temperature (°C)", float(df["MaxTemp"].min()), float(df["MaxTemp"].max()))
humidity = st.slider("Humidity at 9am (%)", 0, 100)
pressure = st.slider("Pressure at 9am (hPa)", int(df["Pressure9am"].min()), int(df["Pressure9am"].max()))
wind_speed = st.slider("Wind Speed at 9am (km/h)", 0, int(df["WindSpeed9am"].max()))

# Encode location
encoded_location = le.transform([location])[0] if location in le.classes_ else 0

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
    rain_prediction = clf.predict(input_data)[0]
    temp_prediction = reg.predict(input_data)[0]

    st.subheader("🌧️ Rain Today Prediction")
    st.write("Yes" if rain_prediction == 1 else "No")

    st.subheader("🌡️ Predicted Average Temperature")
    st.write(f"{temp_prediction:.2f} °C")

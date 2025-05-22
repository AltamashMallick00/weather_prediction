import streamlit as st
import datetime # Just to show some dynamic content

st.set_page_config(layout="wide")
st.title("🚀 Minimal Test App")
st.write(f"This app is working! Current time: {datetime.datetime.now()}")
st.sidebar.header("Test Sidebar")
st.sidebar.text("Hello from sidebar")

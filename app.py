import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
from io import StringIO

st.set_page_config(page_title="NYC Crash Dashboard", layout="wide")
st.title("🚗 NYC Motor Vehicle Collisions Dashboard")

# ---------- Load data from NYC Open Data ----------
@st.cache_data
def load_data():
    url = "https://data.cityofnewyork.us/api/views/h9gi-nx95/rows.csv?accessType=DOWNLOAD"
    
    with st.spinner("🔄 Loading 2M+ crash records from NYC Open Data..."):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            # Use StringIO to handle large data efficiently
            df = pd.read_csv(StringIO(response.text), low_memory=False)
            
            df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'])
            df['CRASH TIME'] = pd.to_datetime(df['CRASH TIME'], format='%H:%M', errors='coerce')
            df['Hour'] = df['CRASH TIME'].dt.hour
            
            return df
        except Exception as e:
            st.error(f"⚠️ Failed to load data: {str(e)}")
            st.info("💡 Please refresh the page. If the problem persists, the NYC Open Data server may be busy.")
            return None

data = load_data()

if data is None:
    st.stop()

# ---------- Sidebar ----------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["📊 Analysis", "🔮 Predict Tomorrow"])

# ---------- Page 1: Analysis ----------
if page == "📊 Analysis":
    st.subheader("📈 Crashes by Hour")
    hourly = data.groupby('Hour').size() / data['Hour'].nunique()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=hourly.index, y=hourly.values, ax=ax, palette="viridis")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Crashes")
    st.pyplot(fig)

    st.subheader("📍 Crashes by Borough")
    borough_counts = data['BOROUGH'].value_counts()
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=borough_counts.index, y=borough_counts.values, ax=ax2, palette="magma")
    ax2.set_xlabel("Borough")
    ax2.set_ylabel("Number of Crashes")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

# ---------- Page 2: Prediction ----------
else:
    st.subheader("🔮 Tomorrow's Crash Prediction")
    tomorrow = datetime.now() + timedelta(days=1)
    st.info(f"📅 Predicting for: **{tomorrow.strftime('%Y-%m-%d')}**")
    st.warning("⚠️ Prediction feature requires model training. Run `2_Prediction_Modeling.ipynb` locally and upload `crash_predictor.pkl` to enable.")
    st.caption("Model: RandomForest | Features: Year, Month, Day, DayOfWeek")

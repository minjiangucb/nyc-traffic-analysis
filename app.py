import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# ---------- Load data directly from NYC Open Data ----------
@st.cache_data
def load_data():
    url = "https://data.cityofnewyork.us/api/views/h9gi-nx95/rows.csv?accessType=DOWNLOAD"
    df = pd.read_csv(url, low_memory=False)
    df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'])
    df['CRASH TIME'] = pd.to_datetime(df['CRASH TIME'], format='%H:%M', errors='coerce')
    df['Hour'] = df['CRASH TIME'].dt.hour
    return df

data = load_data()

# ---------- Page config ----------
st.set_page_config(page_title="NYC Crash Dashboard", layout="wide")
st.title("🚗 NYC Motor Vehicle Collisions Dashboard")

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
    st.info("⚠️ Prediction is available in the local version. Train the model using `2_Prediction_Modeling.ipynb` and upload `crash_predictor.pkl` to enable this feature.")
    tomorrow = datetime.now() + timedelta(days=1)
    st.metric("📅 Date", tomorrow.strftime('%Y-%m-%d'))
    st.caption("Model: RandomForest | Features: Year, Month, Day, DayOfWeek")

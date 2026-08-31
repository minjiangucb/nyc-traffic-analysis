import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime, timedelta

# ---------- Path Configuration ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data/Motor_Vehicle_Collisions_-_Crashes_20250104.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models/crash_predictor.pkl")

# ---------- Page Config ----------
st.set_page_config(page_title="NYC Crash Dashboard", layout="wide")
st.title("🚗 NYC Motor Vehicle Collisions Dashboard")

# ---------- Data Loading ----------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'])
    df['CRASH TIME'] = pd.to_datetime(df['CRASH TIME'], format='%H:%M', errors='coerce')
    df['Hour'] = df['CRASH TIME'].dt.hour
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# ---------- Load Resources ----------
data = load_data()
model = load_model()

# ---------- Sidebar Navigation ----------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["📊 Analysis", "🔮 Predict Tomorrow"])

# ---------- Page 1: Analysis ----------
if page == "📊 Analysis":
    st.subheader("Crashes by Hour")
    hourly = data.groupby('Hour').size() / data['Hour'].nunique()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=hourly.index, y=hourly.values, ax=ax, palette="viridis")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Crashes")
    st.pyplot(fig)

    st.subheader("Crashes by Borough")
    borough_counts = data['BOROUGH'].value_counts()
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=borough_counts.index, y=borough_counts.values, ax=ax2, palette="magma")
    ax2.set_xlabel("Borough")
    ax2.set_ylabel("Number of Crashes")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

# ---------- Page 2: Predict Tomorrow ----------
else:
    st.subheader("🔮 Tomorrow's Crash Prediction")
    tomorrow = datetime.now() + timedelta(days=1)
    st.info(f"Predicting for: **{tomorrow.strftime('%Y-%m-%d')}**")
    
    X_input = pd.DataFrame({
        'Year': [tomorrow.year],
        'Month': [tomorrow.month],
        'Day': [tomorrow.day],
        'DayOfWeek': [tomorrow.weekday()]
    })
    
    pred = int(model.predict(X_input)[0])
    st.metric("Predicted Crashes Tomorrow", pred, delta=None)
    
    st.caption("Model: RandomForest | Features: Year, Month, Day, DayOfWeek")

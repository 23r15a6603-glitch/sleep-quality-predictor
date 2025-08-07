import streamlit as st
import pandas as pd
import joblib
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Load ML model and scaler
model = joblib.load("xgb_sleep_quality_model.pkl")
scaler = joblib.load("scaler_sleep_quality.pkl")

# Streamlit page config
st.set_page_config(page_title="Sleep Quality Predictor", layout="wide")

# Sidebar info
with st.sidebar:
    st.title("😴 Sleep Quality Predictor")
    st.markdown("Predict your sleep quality using AI.\n\nFill out the form 👉")
    st.info("Made with 💙 using XGBoost + DeepSeek R1")

# Title
st.markdown("<h1 style='text-align: center; color: #6C3483;'>AI-Based Sleep Quality Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# Input form
with st.form("sleep_form"):
    st.subheader("📋 Enter Your Health & Lifestyle Data")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 10, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        sleep_duration = st.slider("Sleep Duration (hrs)", 0.0, 12.0, 7.0, 0.5)
        activity = st.slider("Physical Activity (mins/day)", 0, 180, 30)

    with col2:
        stress = st.slider("Stress Level (1–10)", 1, 10, 5)
        caffeine = st.slider("Caffeine Intake (cups/day)", 0, 10, 1)
        alcohol = st.slider("Alcohol Intake (units/day)", 0, 10, 0)
        smoker = st.selectbox("Do you smoke?", ["No", "Yes"])

    with col3:
        heart_rate = st.number_input("Heart Rate (bpm)", 40, 140, 70)
        screen_time = st.slider("Screen Time Before Bed (hrs)", 0.0, 10.0, 2.0, 0.5)
        history = st.selectbox("Sleep Disorder History", ["No", "Yes"])
        bmi = st.number_input("BMI", 10.0, 50.0, 22.0)

    col4, col5 = st.columns(2)
    with col4:
        wake_consistency = st.selectbox("Wake-up Consistency", ["Regular", "Irregular"])
    with col5:
        env_score = st.slider("Sleep Environment Score (1–10)", 1, 10, 7)
        water = st.slider("Daily Water Intake (litres)", 0.0, 5.0, 2.0, 0.5)

    submitted = st.form_submit_button("🔍 Predict Sleep Quality")

# Predict
if submitted:
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [1 if gender == "Male" else 0],
        'Sleep Duration (hrs)': [sleep_duration],
        'Physical Activity (mins/day)': [activity],
        'Stress Level (1–10)': [stress],
        'Caffeine Intake (cups/day)': [caffeine],
        'Alcohol Intake (units/day)': [alcohol],
        'Smoking': [1 if smoker == "Yes" else 0],
        'Heart Rate (bpm)': [heart_rate],
        'Screen Time Before Bed (hrs)': [screen_time],
        'Sleep Disorder History': [1 if history == "Yes" else 0],
        'BMI': [bmi],
        'Wake-up Consistency': [1 if wake_consistency == "Regular" else 0],
        'Sleep Environment Score (1–10)': [env_score],
        'Daily Water Intake (litres)': [water]
    })

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    label_map = {0: 'Poor', 1: 'Fair', 2: 'Good'}
    result = label_map.get(prediction, "Unknown")

    st.success(f"🌙 **Predicted Sleep Quality:** {result}")

# -----------------------------------------------
# 💬 DeepSeek Chatbot via OpenRouter
# -----------------------------------------------
st.markdown("---")
st.subheader("🤖 Ask the DeepSeek Sleep Chatbot")

with st.form("chat_form"):
    prompt = st.text_input("💬 Ask me anything about sleep:")
    send_btn = st.form_submit_button("📨 Send")

if send_btn and prompt:
    if not api_key:
        st.warning("❗ Please set OPENROUTER_API_KEY in your `.env` file.")
    else:
        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://your-app-name.streamlit.app",
                    "X-Title": "Sleep Quality Predictor"
                },
                model="deepseek/deepseek-r1-0528:free",
                messages=[
                    {"role": "system", "content": "You are a helpful AI sleep expert."},
                    {"role": "user", "content": prompt}
                ]
            )

            st.info(completion.choices[0].message.content)

        except Exception as e:
            st.error(f"❌ OpenRouter API Error: {e}")

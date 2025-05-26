# app/main.py
import streamlit as st
import pickle
import pandas as pd

st.title("📰 Fake News Detection App")

vectorizer = pickle.load(open("/home/foxtech/SHAHROZ_PROJ/Fake_news/notebooks/data/processed/vectorizer.pkl", "rb"))
model = pickle.load(open("/home/foxtech/SHAHROZ_PROJ/Fake_news/notebooks/data/processed/model.pkl", "rb"))

user_input = st.text_area("Paste a news article here:")

if st.button("Analyze"):
    vec_input = vectorizer.transform([user_input])
    prediction = model.predict(vec_input)
    label = "🟢 Real News" if prediction[0] == 1 else "🔴 Fake News"
    st.subheader(f"Prediction: {label}")

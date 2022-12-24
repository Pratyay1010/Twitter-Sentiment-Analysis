from main import sentiment_analyzer
import streamlit as st

st.title("Twitter Sentiment Analyzer")

txt = st.text_input('Write a Tweet', '')

if st.button('Analyze'):
    st.write("Sentiment : ",sentiment_analyzer(txt))

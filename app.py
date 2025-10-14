import streamlit as st
import pandas as pd
import io


CPBL_data = pd.read_excel("CPBL_batter.xlsx")
MLB_data = pd.read_excel("MLB_batter.xlsx")
df = pd.concat([MLB_data, CPBL_data], axis=0, ignore_index=True)

st.set_page_config(
    page_title="CPBL & MLB Batter Analysis Main Page",
    layout="wide"
)

st.title("⚾ CPBL & MLB Batter Data Analysis Main Page")

st.title("Dataset Overview")

st.write("This pages, we show the overview of raw data.")


#first row
st.subheader("Dataset Preview")
st.dataframe(df.head())


#info
st.subheader("Dataset Info")
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)


#describe
st.subheader("Statistical Summary")
st.dataframe(df.describe())
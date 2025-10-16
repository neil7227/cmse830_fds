import streamlit as st
import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

#read
CPBL_data = pd.read_excel("CPBL_batter.xlsx")
MLB_data = pd.read_excel("MLB_batter.xlsx")
df = pd.concat([MLB_data, CPBL_data], axis=0, ignore_index=True)

#Streamlit
st.set_page_config(
    page_title="⚾ CPBL & MLB Batter Analysis Main Page",
    layout="wide"
)

st.title("⚾ CPBL & MLB Batter Data Analysis Main Page")
st.write("This page shows the overview of raw data.")

#Dataset Preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

#Info / Heatmap
st.subheader("Dataset Info and Missing Values Overview")

col1, col2 = st.columns([1, 1.3])

with col1:
    st.markdown("#### Dataset Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

with col2:
    st.markdown("#### Missing Values Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False, ax=ax)
    ax.set_title('Missing Values Heatmap', fontsize=12)
    st.pyplot(fig)

#Statistic Summary
st.subheader("Statistical Summary")
st.dataframe(df.describe())

import streamlit as st
import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

#read
CPBL_data_2024 = pd.read_excel("CPBL_batter_2024.xlsx")
CPBL_data_2025 = pd.read_excel("CPBL_batter_2025.xlsx")
CPBL_data = pd.concat([CPBL_data_2024, CPBL_data_2025], axis=0, ignore_index=True)
MLB_data = pd.read_excel("MLB_batter.xlsx")
df = pd.concat([MLB_data, CPBL_data], axis=0, ignore_index=True)

#Streamlit
st.set_page_config(
    page_title="⚾ CPBL & MLB Batter Analysis Main Page",
    layout="wide"
)
tab1, tab2 = st.tabs(["Preview", "Variable Descriptions"])


with tab1:
    st.title("⚾ CPBL & MLB Batter Data Analysis Main Page")
    st.write("Welcome to the CPBL & MLB Batter Data Analysis Dashboard! This dashboard provides an in-depth look at batter performance metrics from both the Chinese Professional Baseball League (CPBL) and Major League Baseball (MLB). Explore various statistical analyses, visualizations, and insights to better understand player performance across these two prominent baseball leagues. This page shows the overview of raw data. I used three datasets: `2025 CPBL batter data`, `2024 CPBL batter data` crawled from `Rebas野球革命`, and `2025 MLB batter data` crawled from `fangraphs` which CPBL is baseball league in Taiwan and MLB is baseball league in the United States. I choose two year of CPBL data is because CPBL play less games in a year.")

    #Dataset Preview
    st.subheader("Original Dataset Preview")
    st.subheader("CPBL 2024 Data Preview")
    st.dataframe(CPBL_data_2024.head())
    st.subheader("CPBL 2025 Data Preview")
    st.dataframe(CPBL_data_2025.head())
    st.subheader("MLB Data Preview")
    st.dataframe(MLB_data.head())
    st.subheader("Combined Dataset Preview")
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

with tab2:
    
    st.title("Variable descriptions")
    st.write("In this page I show the description of each variable in the dataset used for analysis.")
    #variable description
    variable_descriptions = {
        "BB%": "Walk rate — percentage of plate appearances ending in a walk.",
        "K%": "Strikeout rate — percentage of plate appearances ending in a strikeout.",
        "ISO": "Isolated Power — measures extra-base hit power (SLG - AVG).",
        "BABIP": "Batting Average on Balls In Play — how often balls in play go for hits.",
        "AVG": "Batting Average — hits divided by at-bats.",
        "OBP": "On-Base Percentage — times reaching base per plate appearance.",
        "SLG": "Slugging Percentage — total bases per at-bat.",
        "wOBA": "Weighted On-Base Average — overall offensive value per plate appearance.",
        "wRC+": "Weighted Runs Created Plus — offensive value adjusted for park and league (100 = average).",
        "BsR": "Base Running Runs — total baserunning contribution in runs.",
        "Off": "Offensive Runs — total offensive contribution in runs above average.",
        "WAR": "Wins Above Replacement — total player value above a replacement-level player.",
        "OPS+": "On-base Plus Slugging Plus — league- and park-adjusted OPS (100 = average).",
        "PA_scaled": "Scaled Plate Appearances — plate appearances for comparison.",
        "BIP%": "Balls In Play Percentage — rate of contact resulting in fair balls.",
        "PutAway%": "PutAway Percentage — how often a two-strike pitch gets a strikeout.",
        "HR_scaled": "Scaled Home Runs — home run count for fair comparison.",
        "R_scaled": "Scaled Runs — runs scored.",
        "RBI_scaled": "Scaled Runs Batted In — RBI count."
    }

    df_desc = pd.DataFrame(list(variable_descriptions.items()), columns=["Variable", "Description"])
    st.dataframe(df_desc, use_container_width=True)
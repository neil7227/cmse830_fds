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
tab1, tab2 = st.tabs(["Preview", "Variable Descriptions"])


with tab1:
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

with tab2:
    
    st.title("Variable descriptions")

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
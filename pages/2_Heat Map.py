import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Correlation Analysis Dashboard")

df = pd.read_csv("cleaned_data.csv")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
exclude_cols = ['Num', 'Num_Ordi', 'Team_Ordi']
numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

mode = st.sidebar.radio("Mode", ["All Variables", "Correlation with WAR / OPS+"], horizontal=False)
show_values = st.sidebar.checkbox("Show correlation values", value=True)

if mode == "All Variables":
    corr = df[numeric_cols].corr()
else:
    corr = df[numeric_cols].corr()[['WAR', 'OPS+']].sort_values(by='WAR', ascending=False)

fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(
    corr,
    annot=show_values,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={"shrink": 0.7},
    ax=ax
)

if mode == "All Variables":
    ax.set_title("Correlation Heatmap (All Variables)", fontsize=16, pad=15)
else:
    ax.set_title("Correlation with WAR / OPS+", fontsize=16, pad=15)

st.pyplot(fig)

st.markdown("---")

st.markdown("<h3 style='color:white;'>Top 5 Features Correlated with Selected Variables</h3>", unsafe_allow_html=True)

var1 = st.sidebar.selectbox("Select first variable", numeric_cols, index=numeric_cols.index('WAR'))
var2 = st.sidebar.selectbox("Select second variable", numeric_cols, index=numeric_cols.index('OPS+'))

corr_matrix = df[numeric_cols].corr()

top_var1 = corr_matrix[var1].abs().sort_values(ascending=False).drop(var1).head(5)
top_var2 = corr_matrix[var2].abs().sort_values(ascending=False).drop(var2).head(5)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**Top 5 correlated with {var1}**")
    st.dataframe(
        top_var1.reset_index().rename(columns={'index': 'Variable', var1: 'Correlation'}).style.format({"Correlation": "{:.2f}"})
    )

with col2:
    st.markdown(f"**Top 5 correlated with {var2}**")
    st.dataframe(
        top_var2.reset_index().rename(columns={'index': 'Variable', var2: 'Correlation'}).style.format({"Correlation": "{:.2f}"})
    )

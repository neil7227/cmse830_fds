import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

st.title("Correlation Analysis Dashboard")

CPBL_data = pd.read_excel("CPBL_batter.xlsx")
MLB_data = pd.read_excel("MLB_batter.xlsx")
df_bef = pd.concat([CPBL_data, MLB_data], axis=0, ignore_index=True)
df = pd.read_csv("cleaned_data.csv")

# --- Data cleaning ---
CPBL_Game, MLB_Game = 120, 162
CPBL_data.drop(columns=['BB/K','OPS','tOPS+','RC'], inplace=True)
MLB_data.drop(columns=['G','xwOBA','Def','SB'], inplace=True)

CPBL_data.rename(columns={'球員':'Name','背號':'Num','隊伍':'Team'}, inplace=True)
MLB_data[['K%','BB%']] = MLB_data[['K%','BB%']] * 100
CPBL_data['PA_scaled'] = CPBL_data['PA']/CPBL_Game
MLB_data['PA_scaled'] = MLB_data['PA']/MLB_Game

df_bef = pd.concat([MLB_data, CPBL_data], axis=0, ignore_index=True)
df_bef = df_bef[df_bef['PA_scaled'] > 1]
Scale = 600
df_bef['HR_scaled'] = df_bef['HR'] * Scale / df_bef['PA']
df_bef['R_scaled'] = df_bef['R'] * Scale / df_bef['PA']
df_bef['RBI_scaled'] = df_bef['RBI'] * Scale / df_bef['PA']
df_bef.drop(columns=['HR','R','RBI','PA'], inplace=True)

# --- Encoding ---
st.subheader("Encoding")
oe = OrdinalEncoder()
df_bef['Num_Ordi'] = oe.fit_transform(df_bef[['Num']])
df_bef['Team_Ordi'] = oe.fit_transform(df_bef[['Team']])

# --- Numeric DataFrame ---
num_col = df_bef.select_dtypes(include=[np.number]).columns.tolist()
df_num = df_bef[num_col].copy()
for c in ['Num','Num_Ordi','Team_Ordi']:
    if c in df_num.columns:
        df_num.drop(columns=c, inplace=True)

# --- Split league ---
df_num["CPBL"] = False
df_num.loc[df_num["HR_scaled"].isnull()==True, "CPBL"] = True

# numeric
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
exclude_cols = ['Num', 'Num_Ordi', 'Team_Ordi']
numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

# --- data switch ---
st.markdown("### Choose which dataset to visualize")
dataset_choice = st.radio(
    "Select dataset:",
    ("Cleaned Data", "Before Cleaning"),
    horizontal=True
)
# switch DataFrame
if dataset_choice == "Before Cleaning":
    current_df = df_bef
else:
    current_df = df


numeric_cols_current = current_df.select_dtypes(include=['float64', 'int64']).columns
numeric_cols_current = [col for col in numeric_cols_current if col not in exclude_cols]

# --- side bar ---
mode = st.sidebar.radio("Mode", ["All Variables", "Correlation with WAR / OPS+"], horizontal=False)
show_values = st.sidebar.checkbox("Show correlation values", value=True)

# --- calculate ---
if mode == "All Variables":
    corr = current_df[numeric_cols_current].corr()
else:
    if 'WAR' in current_df.columns and 'OPS+' in current_df.columns:
        corr = current_df[numeric_cols_current].corr()[['WAR', 'OPS+']].sort_values(by='WAR', ascending=False)
    else:
        st.warning("WAR or OPS+ not found in current dataset.")
        st.stop()

# --- heatmap ---
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

# target set
if mode == "All Variables":
    ax.set_title(f"Correlation Heatmap ({dataset_choice})", fontsize=16, pad=15)
else:
    ax.set_title(f"Correlation with WAR / OPS+ ({dataset_choice})", fontsize=16, pad=15)

st.pyplot(fig)

# --- Top correlated features ---
st.markdown("---")
st.markdown("### 🔝 Top 5 Features Correlated with Selected Variables")

var1 = st.sidebar.selectbox("Select first variable", numeric_cols_current, index=min(len(numeric_cols_current)-1, numeric_cols_current.index('WAR') if 'WAR' in numeric_cols_current else 0))
var2 = st.sidebar.selectbox("Select second variable", numeric_cols_current, index=min(len(numeric_cols_current)-1, numeric_cols_current.index('OPS+') if 'OPS+' in numeric_cols_current else 0))

corr_matrix = current_df[numeric_cols_current].corr()

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

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

df = pd.read_csv("cleaned_data.csv")

st.title("Player Performance Overview")

stats = {
    'WAR': {
        'sum': round(df['WAR'].sum(),2),
        'mean': round(df['WAR'].mean(),2),
        'median': round(df['WAR'].median(),2),
        'mode': round(df['WAR'].mode()[0],2)
    },
    'OPS+': {
        'sum': round(df['OPS+'].sum(),2),
        'mean': round(df['OPS+'].mean(),2),
        'median': round(df['OPS+'].median(),2),
        'mode': round(df['OPS+'].mode()[0],2)
    }
}

st.subheader("WAR & OPS+ Statistics")
for metric in ['WAR', 'OPS+']:
    st.markdown(f"### {metric}", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, stat in enumerate(['sum','mean','median','mode']):
        with cols[i]:
            st.markdown(f"<div style='background-color:#0a3d62; color:white; padding:10px; border-radius:5px; text-align:center;'>"
                        f"<div style='font-size:14px'>{stat.upper()}</div>"
                        f"<div style='font-size:28px'>{stats[metric][stat]}</div>"
                        f"</div>", unsafe_allow_html=True)

st.subheader("Distribution Charts")
col1, col2 = st.columns(2)

with col1:
    fig_war, ax = plt.subplots()
    sns.histplot(df['WAR'], kde=True, ax=ax, color='skyblue', bins=20)
    ax.set_title("WAR Distribution", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_war)

with col2:
    fig_ops, ax = plt.subplots()
    sns.histplot(df['OPS+'], kde=True, ax=ax, color='lightgreen', bins=20)
    ax.set_title("OPS+ Distribution", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_ops)


st.subheader("Team Distribution")
team_counts = df['Team'].value_counts()
colors = sns.color_palette('tab20', n_colors=len(team_counts))


font_path = "C:/Windows/Fonts/msjh.ttc"  
myfont = font_manager.FontProperties(fname=font_path)

fig_pie, ax = plt.subplots(figsize=(12,10))
wedges, texts, autotexts = ax.pie(
    team_counts, 
    labels=team_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    textprops={'fontsize':6, 'fontproperties': myfont},
    wedgeprops={'edgecolor':'white'},
    pctdistance=0.85,     
    labeldistance=1.05    
)
ax.set_title("Team Distribution", fontsize=14)
st.pyplot(fig_pie)

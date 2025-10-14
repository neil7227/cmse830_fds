import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

st.title("Visualization Dashboard")

df = pd.read_csv("cleaned_data.csv")
numeric_cols = [x for x in df.select_dtypes(include=np.number).columns if x not in ['Num_Ordi', 'Team_Ordi']]
teams = df['Team'].unique().tolist()
teams.insert(0, "All")

selected_team = st.sidebar.multiselect("Select teams to include", teams, default=["All"])
var_x = st.sidebar.selectbox("Select X variable (scatter)", numeric_cols, index=numeric_cols.index("WAR") if "WAR" in numeric_cols else 0)
var_y = st.sidebar.selectbox("Select Y variable (scatter)", ["All"] + numeric_cols, index=0)
hue_option = st.sidebar.checkbox("Color by Team", value=True)
hist_var = st.sidebar.selectbox("Select variable for histogram", ["All"] + numeric_cols, index=0)
violin_var = st.sidebar.selectbox("Select variable for violin plot", numeric_cols, index=numeric_cols.index("OPS+") if "OPS+" in numeric_cols else 0)

if "All" not in selected_team:
    df_plot = df[df['Team'].isin(selected_team)]
else:
    df_plot = df.copy()

st.subheader("Histograms")
if hist_var == "All":
    n_cols = 5
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*2.5))
    axes = axes.flatten()
    for i, var in enumerate(numeric_cols):
        sns.histplot(df_plot[var], kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(var, fontsize=8)
        axes[i].tick_params(axis='x', labelsize=7, rotation=45)
        axes[i].tick_params(axis='y', labelsize=7)
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    st.pyplot(fig)
else:
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df_plot[hist_var], kde=True, ax=ax, color='skyblue')
    ax.set_title(f"Histogram of {hist_var}")
    plt.tight_layout()
    st.pyplot(fig)

st.subheader("Scatter Plot")
if var_y == "All":
    scatter_vars = [v for v in numeric_cols if v != var_x]
    n_cols = 5
    n_rows = int(np.ceil(len(scatter_vars)/n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
    axes = axes.flatten()
    palette = sns.color_palette("tab20", n_colors=df_plot['Team'].nunique())
    for i, y_var in enumerate(scatter_vars):
        sns.scatterplot(
            data=df_plot,
            x=var_x,
            y=y_var,
            hue='Team' if hue_option else None,
            s=50,
            alpha=0.7,
            palette=palette,
            ax=axes[i]
        )
        axes[i].set_title(f"{y_var} vs {var_x}", fontsize=10)
        axes[i].tick_params(labelsize=8)
        legend = axes[i].get_legend()
        if hue_option and (i % n_cols == n_cols-1):
            axes[i].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=2, fontsize=8)
        elif legend is not None:
            legend.remove()
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    st.pyplot(fig)
else:
    y_var = var_y
    fig, ax = plt.subplots(figsize=(6,4))
    if hue_option:
        sns.scatterplot(data=df_plot, x=var_x, y=y_var, hue='Team', palette='tab20', ax=ax)
        ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', ncol=2, fontsize=8)
    else:
        sns.scatterplot(data=df_plot, x=var_x, y=y_var, ax=ax, color='skyblue')
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    ax.set_title(f"{y_var} vs {var_x}")
    plt.tight_layout()
    st.pyplot(fig)

st.subheader("Violin Plot")
plt.figure(figsize=(24,10))
if hue_option:
    sns.violinplot(
        data=df_plot,
        x='Team',
        y=violin_var,
        palette='Set2',
        inner='box',
        width=0.9,
        hue='Team'
    )
else:
    sns.violinplot(
        data=df_plot,
        x='Team',
        y=violin_var,
        palette='Set2',
        inner='box',
        width=0.9
    )
plt.xticks(rotation=90)
plt.xlabel('Team', fontsize=12)
plt.ylabel(violin_var, fontsize=12)
plt.title(f'{violin_var}/Team', fontsize=16)
plt.tight_layout()
st.pyplot(plt)

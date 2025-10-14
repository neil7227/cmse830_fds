import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
import io

# --- Streamlit config ---
st.set_page_config(page_title="CPBL & MLB IDA/EDA", layout="wide")

st.title("⚾ CPBL & MLB Data Processing Visualization")

# --- Load data ---
CPBL_data = pd.read_excel("CPBL_batter.xlsx")
MLB_data = pd.read_excel("MLB_batter.xlsx")
df = pd.concat([MLB_data, CPBL_data], axis=0, ignore_index=True)

# --- Compare columns ---
same_col = set(CPBL_data.columns) & set(MLB_data.columns)
diff_col = set(CPBL_data.columns) ^ set(MLB_data.columns)
all_columns = df.columns.tolist()

st.subheader("Dataset Columns Overview")
st.text(
    f"All Columns ({len(all_columns)}): {', '.join(all_columns)}\n\n"
    f"Same Columns ({len(same_col)}): {', '.join(same_col)}\n\n"
    f"Different Columns ({len(diff_col)}): {', '.join(diff_col)}"
)

# --- Data Cleaning ---
Scale = 1.232
CPBL_Game = 120
MLB_Game = 162

CPBL_data.drop(columns=['BB/K','OPS','tOPS+','RC'], inplace=True)
MLB_data.drop(columns=['G','xwOBA','Def','SB'], inplace=True)

MLB_data['OPS+'] = 100 * (MLB_data['OBP']/MLB_data['OBP'].mean() +
                           MLB_data['SLG']/MLB_data['SLG'].mean() - 1)
CPBL_data['Off'] = ((CPBL_data['wOBA'] / CPBL_data['wOBA'].mean()) - 1) * 100
CPBL_data.rename(columns={'球員': 'Name', '背號': 'Num', '隊伍': 'Team'}, inplace=True)
MLB_data[['K%', 'BB%']] = MLB_data[['K%', 'BB%']] * 100
CPBL_data['PA_scaled'] = CPBL_data['PA']/CPBL_Game
MLB_data['PA_scaled'] = MLB_data['PA']/MLB_Game

df = pd.concat([MLB_data, CPBL_data], axis=0, ignore_index=True)
df = df[df['PA_scaled'] > 1]
Scale = 600
df['HR_scaled'] = df['HR'] * Scale / df['PA']
df['R_scaled'] = df['R'] * Scale / df['PA']
df['RBI_scaled'] = df['RBI'] * Scale / df['PA']
df.drop(columns=['HR','R','RBI','PA'], inplace=True)

st.subheader("Dataset Preview after initial cleaning")
st.dataframe(df.head())

# --- Encoding ---
st.subheader("Encoding")
oe = OrdinalEncoder()
df['Num_Ordi'] = oe.fit_transform(df[['Num']])
df['Team_Ordi'] = oe.fit_transform(df[['Team']])
st.write("Ordinal encoding applied to 'Num' and 'Team':")
st.dataframe(df[['Num','Num_Ordi','Team','Team_Ordi']].head())

# --- Numeric DataFrame ---
num_col = df.select_dtypes(include=[np.number]).columns.tolist()
df_num = df[num_col].copy()
for c in ['Num','Num_Ordi','Team_Ordi']:
    if c in df_num.columns:
        df_num.drop(columns=c, inplace=True)

# --- Split league ---
df_num["CPBL"] = False
df_num.loc[df_num["HR_scaled"].isnull() == True, "CPBL"] = True

# --- Scaling ---
st.subheader("Scaling (RobustScaler)")
num_cols = df_num.select_dtypes(include='number').columns.tolist()
num_cols = [c for c in num_cols if c != 'CPBL']
scaler = RobustScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_num[num_cols]),
                         columns=num_cols, index=df_num.index)
st.dataframe(df_scaled.head())

# --- Histograms of scaled data ---
st.subheader("Histograms of Scaled Numeric Data")
fig, axes = plt.subplots(5, 4, figsize=(20,16))
axes = axes.flatten()
for i, c in enumerate(num_cols):
    sns.histplot(df_scaled[c], bins=20, alpha=0.6, ax=axes[i])
    axes[i].set_title(c, fontsize=10)
for j in range(i+1,len(axes)):
    axes[j].axis('off')
plt.tight_layout()
st.pyplot(fig)

# --- Imputation ---
st.subheader("Imputation using IterativeImputer")
feature = ['ISO','SLG','OPS+','HR_scaled']
imputer = IterativeImputer(random_state=42, sample_posterior=True, max_iter=10)
df_imputed = df_scaled.copy()
df_imputed[feature] = imputer.fit_transform(df_scaled[feature])
st.dataframe(df_imputed[feature].head())

# --- Linear Regression for missing wRC+ ---
st.subheader("Linear Regression for missing wRC+")
df_clean = df_imputed.dropna(subset=['wRC+'])
X_train = df_clean[['Off','OPS+','wOBA']]
y_train = df_clean['wRC+']
linear_model = LinearRegression().fit(X_train, y_train)
missing = df_imputed.loc[df_imputed['wRC+'].isna(), ['Off','OPS+','wOBA']]
df_imputed.loc[df_imputed['wRC+'].isna(), 'wRC+'] = linear_model.predict(missing)
st.write("wRC+ after regression imputation:")
st.dataframe(df_imputed[['Off','OPS+','wOBA','wRC+']].head())

# --- Linear Regression for missing R_scaled ---
st.subheader("Linear Regression for missing R_scaled")
df_clean = df_imputed.dropna(subset=['R_scaled'])
X_train = df_clean[['Off','wRC+','wOBA']]
y_train = df_clean['R_scaled']
linear_model = LinearRegression().fit(X_train, y_train)
missing = df_imputed.loc[df_imputed['R_scaled'].isna(), ['Off','wRC+','wOBA']]
df_imputed.loc[df_imputed['R_scaled'].isna(), 'R_scaled'] = linear_model.predict(missing)
st.write("R_scaled after regression imputation:")
st.dataframe(df_imputed[['Off','wRC+','wOBA','R_scaled']].head())

# --- Final DataFrame ---
df_num = df_imputed.copy()
df_num = df_num[df_num['PA_scaled'] >= 3]
df.update(df_num)
df = df[df['PA_scaled'] >= 3]

st.subheader("Final Dataset after Scaling & Imputation")
st.dataframe(df.head())
st.write(df.describe())

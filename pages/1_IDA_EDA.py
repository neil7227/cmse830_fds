import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# --- Streamlit config ---
st.set_page_config(page_title="CPBL & MLB IDA/EDA", layout="wide")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

st.title("⚾ CPBL & MLB Data Processing Visualization")
tab1, tab2, tab3 = st.tabs(["Preview/Encoding/Scaling", "Heat Map", "Imputing"])

with tab1:
    # --- Load data ---
    CPBL_data = pd.read_excel("CPBL_batter.xlsx")
    MLB_data = pd.read_excel("MLB_batter.xlsx")
    df = pd.concat([CPBL_data, MLB_data], axis=0, ignore_index=True)

    # --- Data cleaning ---
    CPBL_Game, MLB_Game = 120, 162
    CPBL_data.drop(columns=['BB/K','OPS','tOPS+','RC'], inplace=True)
    MLB_data.drop(columns=['G','xwOBA','Def','SB'], inplace=True)

    MLB_data['OPS+'] = 100 * (MLB_data['OBP']/MLB_data['OBP'].mean() +
                            MLB_data['SLG']/MLB_data['SLG'].mean() - 1)
    CPBL_data['Off'] = ((CPBL_data['wOBA']/CPBL_data['wOBA'].mean()) - 1) * 100
    CPBL_data.rename(columns={'球員':'Name','背號':'Num','隊伍':'Team'}, inplace=True)
    MLB_data[['K%','BB%']] = MLB_data[['K%','BB%']] * 100
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
    st.dataframe(df[['Num','Num_Ordi','Team','Team_Ordi']].dropna(subset=['Num','Team']).head(10))

    # --- Numeric DataFrame ---
    num_col = df.select_dtypes(include=[np.number]).columns.tolist()
    df_num = df[num_col].copy()
    for c in ['Num','Num_Ordi','Team_Ordi']:
        if c in df_num.columns:
            df_num.drop(columns=c, inplace=True)

    # --- Split league ---
    df_num["CPBL"] = False
    df_num.loc[df_num["HR_scaled"].isnull()==True, "CPBL"] = True

    # --- Scaling ---
    st.subheader("Scaling (RobustScaler)")
    num_cols = [c for c in df_num.select_dtypes(include='number').columns if c != 'CPBL']
    scaler = RobustScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_num[num_cols]), columns=num_cols, index=df_num.index)
    st.dataframe(df_scaled.head())
with tab2:
    # --- Heatmap ---
    st.subheader("Heatmap of Numeric Features")
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(df_scaled.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)
with tab3:
    # --- Iterative Imputer ---
    st.subheader("Iterative Imputer")
    impute_groups = [
        ['ISO','SLG','OPS+','HR_scaled'],
        ['ISO','SLG','RBI_scaled','HR_scaled'],
        ['K%','BIP%','PutAway%','PA_scaled','AVG']
    ]

    df_imputed = df_scaled.copy()
    for feature in impute_groups:
        imputer = IterativeImputer(random_state=42, sample_posterior=True, max_iter=10)
        df_imputed[feature] = imputer.fit_transform(df_imputed[feature])
        st.write(f"Imputed features: {feature}")
        st.dataframe(df_imputed[feature].head(10))


    # --- Linear Regression Imputer for wRC+ ---
    st.subheader("Linear Regression Imputation: wRC+")
    df_clean = df_imputed.dropna(subset=['wRC+'])
    lr_wRC = LinearRegression().fit(df_clean[['Off','OPS+','wOBA']], df_clean['wRC+'])
    missing = df_imputed.loc[df_imputed['wRC+'].isna(), ['Off','OPS+','wOBA']]
    df_imputed.loc[df_imputed['wRC+'].isna(), 'wRC+'] = lr_wRC.predict(missing)
    st.dataframe(df_imputed[['Off','OPS+','wOBA','wRC+']].head())

    # --- Linear Regression Imputer for R_scaled ---
    st.subheader("Linear Regression Imputation: R_scaled")
    df_clean = df_imputed.dropna(subset=['R_scaled'])
    lr_R = LinearRegression().fit(df_clean[['Off','wRC+','wOBA']], df_clean['R_scaled'])
    missing = df_imputed.loc[df_imputed['R_scaled'].isna(), ['Off','wRC+','wOBA']]
    df_imputed.loc[df_imputed['R_scaled'].isna(), 'R_scaled'] = lr_R.predict(missing)
    st.dataframe(df_imputed[['Off','wRC+','wOBA','R_scaled']].head())

    # --- RandomForest Imputer ---
    st.subheader("RandomForest Imputer: R_scaled, WAR, BsR")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=3,
                            random_state=42, n_jobs=-1)
    rf_imputer = IterativeImputer(estimator=rf, random_state=42, max_iter=70, sample_posterior=False)
    df_imputed[['R_scaled','WAR','BsR']] = rf_imputer.fit_transform(df_imputed[['R_scaled','WAR','BsR']])
    st.dataframe(df_imputed[['R_scaled','WAR','BsR']].head())

    # --- Final KNN Imputer ---
    st.subheader("Final KNN Imputer")
    knn_imputer = KNNImputer(n_neighbors=1)
    df_final = pd.DataFrame(knn_imputer.fit_transform(df_imputed), columns=df_imputed.columns)
    st.dataframe(df_final.head())

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

st.title("⚾ CPBL & MLB Data Processing")
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
    st.dataframe(df[['Num','Num_Ordi','Team']].dropna(subset=['Num','Team']).head(10))

    # --- Numeric DataFrame ---
    num_col = df.select_dtypes(include=[np.number]).columns.tolist()
    df_num = df[num_col].copy()
    for c in ['Num','Num_Ordi']:
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
    df_imputed = df_scaled.copy()
    df_imputed['Num_Ordi'] = df['Num_Ordi']
    # --- Linear Regression Imputer for wRC+ ---
    df_imputed['was_missing'] = df['wRC+'].isna()
    st.subheader("Linear Regression Imputation: wRC+")
    df_clean = df_imputed.dropna(subset=['wRC+'])
    lr_wRC = LinearRegression().fit(df_clean[['Off','OPS+','wOBA']], df_clean['wRC+'])
    missing = df_imputed.loc[df_imputed['wRC+'].isna(), ['Off','OPS+','wOBA']]
    df_imputed.loc[df_imputed['wRC+'].isna(), 'wRC+'] = lr_wRC.predict(missing)
    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(df_imputed['wRC+'].describe())

    with col2:
        fig, ax = plt.subplots(figsize=(6,3))
        for label, group_data in df_imputed.groupby('was_missing'):
            sns.kdeplot(data=group_data, x='wRC+', fill=True, label=f"Was Missing = {label}", ax=ax)
            
            ax.set_title('wRC+ Distribution (Imputed vs Original)')
            ax.set_xlabel('wRC+')
            ax.set_ylabel('Density')
            ax.legend(title='Missing Status')
            
        st.pyplot(fig)

    # --- Linear Regression Imputer for R_scaled ---
    df_imputed['was_missing'] = df['R_scaled'].isna()
    st.subheader("Linear Regression Imputation: R_scaled")
    df_clean = df_imputed.dropna(subset=['R_scaled'])
    lr_R = LinearRegression().fit(df_clean[['Off','wRC+','wOBA']], df_clean['R_scaled'])
    missing = df_imputed.loc[df_imputed['R_scaled'].isna(), ['Off','wRC+','wOBA']]
    df_imputed.loc[df_imputed['R_scaled'].isna(), 'R_scaled'] = lr_R.predict(missing)
    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(df_imputed['R_scaled'].describe())
    with col2:
        fig, ax = plt.subplots(figsize=(6,3))
        for label, group_data in df_imputed.groupby('was_missing'):
            sns.kdeplot(data=group_data, x='R_scaled', fill=True, label=f"Was Missing = {label}", ax=ax)
            
            ax.set_title('R_scaled Distribution (Imputed vs Original)')
            ax.set_xlabel('R_scaled')
            ax.set_ylabel('Density')
            ax.legend(title='Missing Status')
            
        st.pyplot(fig)


    # --- Iterative Imputer ---
    st.subheader("Iterative Imputer")
    impute_groups = [
        ['ISO','SLG','OPS+','HR_scaled'],
        ['ISO','SLG','RBI_scaled','HR_scaled'],
        ['K%','BIP%','PutAway%','PA_scaled','AVG'],
        ['PA_scaled', 'BB%', 'ISO', 'BABIP', 'AVG', 'OBP', 'SLG', 'wOBA', 'wRC+', 'Off', 'WAR', 'OPS+', 'HR_scaled', 'R_scaled']
    ]
    target_col = ['HR_scaled', 'RBI_scaled', 'PutAway%', 'BIP%', 'WAR']

    for i,feature in enumerate(impute_groups):
        if i < 3:
            df_imputed['was_missing'] = df[target_col[i]].isna()
        else:
            df_imputed['was_missing'] = df[target_col[i+1]].isna()
        imputer = IterativeImputer(random_state=42, sample_posterior=True, max_iter=10)
        df_imputed[feature] = imputer.fit_transform(df_imputed[feature])
        col1, col2 = st.columns(2)
        with col1:
            if i < 2:
                st.subheader(f"Target features: {target_col[i]}")
                st.write(f"Imputed features: {feature}")
                st.dataframe(df_imputed[target_col[i]].describe())
            elif i < 3:
                st.subheader(f"Target features: {target_col[i]}, {target_col[i+1]}")
                st.write(f"Imputed features: {feature}")
                st.dataframe(df_imputed[target_col[i]].describe())
                st.dataframe(df_imputed[target_col[i+1]].describe())
            else: 
                st.subheader(f"Target features: {target_col[i+1]}")
                st.write(f"Imputed features: {feature}")
                st.dataframe(df_imputed[target_col[i+1]].describe())
        with col2:
            if i < 2:
                fig, ax = plt.subplots(figsize=(6,3))
                for label, group_data in df_imputed.groupby('was_missing'):
                    sns.kdeplot(data=group_data, x=target_col[i], fill=True, label=f"Was Missing = {label}", ax=ax)
                    
                    ax.set_title(f'{target_col[i]} Distribution (Imputed vs Original)')
                    ax.set_xlabel(target_col[i])
                    ax.set_ylabel('Density')
                    ax.legend(title='Missing Status')
                    
                st.pyplot(fig)
            elif i < 3:
                fig, ax = plt.subplots(figsize=(6,3))
                for label, group_data in df_imputed.groupby('was_missing'):
                    sns.kdeplot(data=group_data, x=target_col[i], fill=True, label=f"Was Missing = {label}", ax=ax)
                    
                    ax.set_title(f'{target_col[i]} Distribution (Imputed vs Original)')
                    ax.set_xlabel(target_col[i])
                    ax.set_ylabel('Density')
                    ax.legend(title='Missing Status')
                st.pyplot(fig)
                
                fig, ax = plt.subplots(figsize=(6,3))
                for label, group_data in df_imputed.groupby('was_missing'):
                    sns.kdeplot(data=group_data, x=target_col[i+1], fill=True, label=f"Was Missing = {label}", ax=ax)

                    ax.set_title(f'{target_col[i+1]} Distribution (Imputed vs Original)')
                    ax.set_xlabel(target_col[i+1])
                    ax.set_ylabel('Density')
                    ax.legend(title='Missing Status')
                st.pyplot(fig)
            else: 
                fig, ax = plt.subplots(figsize=(6,3))
                for label, group_data in df_imputed.groupby('was_missing'):
                    sns.kdeplot(data=group_data, x=target_col[i+1], fill=True, label=f"Was Missing = {label}", ax=ax)
                    
                    ax.set_title(f'{target_col[i+1]} Distribution (Imputed vs Original)')
                    ax.set_xlabel(target_col[i+1])
                    ax.set_ylabel('Density')
                    ax.legend(title='Missing Status')
                st.pyplot(fig)
                
    # --- RandomForest Imputer ---
    st.subheader("RandomForest Imputer: BsR")
    df_imputed['was_missing'] = df['BsR'].isna()
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=3,
                            random_state=42, n_jobs=-1)
    rf_imputer = IterativeImputer(estimator=rf, random_state=42, max_iter=70, sample_posterior=False)
    df_imputed[['R_scaled','WAR','BsR']] = rf_imputer.fit_transform(df_imputed[['R_scaled','WAR','BsR']])
    
    with col1:
        st.dataframe(df_imputed['BsR'].describe())
    with col2:
        fig, ax = plt.subplots(figsize=(6,3))
        for label, group_data in df_imputed.groupby('was_missing'):
            sns.kdeplot(data=group_data, x='BsR', fill=True, label=f"Was Missing = {label}", ax=ax)
            
            ax.set_title('BsR Distribution (Imputed vs Original)')
            ax.set_xlabel('BsR')
            ax.set_ylabel('Density')
            ax.legend(title='Missing Status')
            
        st.pyplot(fig)

    # --- Final KNN Imputer ---
    st.subheader("KNN Imputer: Num")
    df_imputed['was_missing'] = df['Num_Ordi'].isna()
    knn_imputer = KNNImputer(n_neighbors=1)
    df_imputed = pd.DataFrame(knn_imputer.fit_transform(df_imputed), columns=df_imputed.columns)
    num_categories = len(oe.categories_[0])
    df_imputed['Num_Ordi'] = df_imputed['Num_Ordi'].round().clip(0, num_categories-1).astype(int)
    df_imputed['Num'] = oe.inverse_transform(df_imputed[['Num_Ordi']])[:, 0]

    df_imputed['Num_plot'] = df_imputed['Num'].astype(int).astype(str)  
    fig, ax = plt.subplots(figsize=(8,4))
    sns.countplot(data=df_imputed, x='Num_plot', hue='was_missing', ax=ax)
    ax.set_title('Num Distribution (Imputed vs Original)', fontsize=12)
    ax.set_xlabel('Num', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.legend(title='Was Missing', fontsize=9, title_fontsize=10)
    ax.tick_params(axis='x', rotation=90, labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)


    
    df_imputed.drop(columns = 'was_missing', inplace = True)
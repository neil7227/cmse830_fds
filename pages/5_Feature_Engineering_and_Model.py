import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import statsmodels.api as sm

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

tab1, tab2 = st.tabs(["PCA", "Model Preparation"])

with tab1:
    st.title("PCA Feature Engineering")
    st.write("This section I do one hot encoding to make sure categorical variables are properly handled.")

    # --- 讀取資料 ---
    df = pd.read_csv("cleaned_data.csv")

    # --- One-Hot Encoding: Team ---
    st.subheader("One-Hot Encoding: Team")
    df_team_ohe = pd.get_dummies(df['Team'], prefix='Team')
    st.dataframe(df_team_ohe.head(10))

    # --- One-Hot Encoding: Num ---
    st.subheader("One-Hot Encoding: Num_Ordi")
    df['Num_Ordi'] = df['Num_Ordi'].astype(str)
    df_num_ohe = pd.get_dummies(df['Num_Ordi'], prefix='Num_Ordi')
    st.dataframe(df_num_ohe.head(10))

    # --- 合併數值特徵 + One-Hot ---
    feature_cols = [col for col in df.columns if col not in ['WAR', 'Num', 'Team', 'Num_Ordi', 'Team_Ordi', 'Name']]
    X = pd.concat([
        df[feature_cols].reset_index(drop=True),
        df_team_ohe.reset_index(drop=True),
        df_num_ohe.reset_index(drop=True)
    ], axis=1)

    y = df['WAR']

    # --- 標準化 ---
    X_numeric = X.select_dtypes(include=['float64', 'int64'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    # --- PCA ---
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    # --- 顯示 PCA 結果 ---
    st.subheader("PCA Results")
    st.write("In this section, I applied PCA to reduce the dimensionality of the dataset while retaining 95% of the variance.")
    st.write(f"Original number of features: **{X.shape[1]}**")
    st.write(f"Number of principal components retained: **{X_pca.shape[1]}**")
    st.write("### Explained Variance Ratio (per component)")
    st.write(pca.explained_variance_ratio_)
    st.write("### Cumulative Explained Variance Ratio")
    st.write(pca.explained_variance_ratio_.cumsum())
    
    # --- SVD for Biplot ---
    U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    V = Vt.T

    # --- Variance explained ---
    var_exp = s**2 / np.sum(s**2)
    cum_var_exp = np.cumsum(var_exp)

    # --- Loadings Table ---
    st.subheader("Feature Loadings (PC1 & PC2)")
    loadings = pd.DataFrame({
        'Feature': X_numeric.columns,
        'PC1': V[:, 0] * s[0],
        'PC2': V[:, 1] * s[1]
    })
    st.dataframe(loadings)

    fig = plt.figure(figsize=(10, 12))

    st.write("""### PCA Scree Plot and Biplot
    The scree plot shows the proportion of variance explained by each principal component, helping to visualize how many components to retain.
    The biplot displays both the scores of the samples and the loadings of the features on the first two principal components.
    """)
    
    # 1. Scree Plot
    plt.subplot(211)
    plt.plot(range(1, len(var_exp) + 1), var_exp, 'bo-', label='Individual')
    plt.plot(range(1, len(cum_var_exp) + 1), cum_var_exp, 'ro-', label='Cumulative')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.title('Scree Plot')
    plt.grid(True)
    plt.legend()
    
    # 2. Biplot
    plt.subplot(212)
    scores = X_scaled @ V
    scale = 3

    plt.scatter(scores[:, 0], scores[:, 1], c='blue', alpha=0.5)

    for i, feature in enumerate(X_numeric.columns):
        x = V[i, 0] * s[0] * scale
        y_arrow = V[i, 1] * s[1] * scale  # 注意不要覆蓋 y
        plt.arrow(0, 0, x, y_arrow, color='red', alpha=0.5, head_width=0.03)

        ha = 'left' if x >= 0 else 'right'
        va = 'bottom' if y_arrow >= 0 else 'top'
        plt.text(x * 1.1, y_arrow * 1.1, feature, ha=ha, va=va)

    plt.xlabel(f"PC1 ({var_exp[0]:.1%})")
    plt.ylabel(f"PC2 ({var_exp[1]:.1%})")
    plt.title("PCA Biplot")
    plt.grid(True)

    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.title("Model Preparation")
    st.write("This section builds machine learning models using original features or PCA-transformed features.")

    # --- Sidebar: PCA components selection ---
    st.sidebar.header("PCA Settings")
    pca_n = st.sidebar.slider(
        "Number of PCA components to use",
        min_value=1,
        max_value=X_scaled.shape[1],
        value=min(5, X_scaled.shape[1]),   # default = 5 or max
        step=1
    )

    # --- 選擇是否使用 PCA ---
    use_pca = st.radio(
        "Use PCA-transformed features?",
        ("No (use original scaled features)", "Yes (use PCA features)")
    )

    # --- Feature selection ---
    if use_pca == "Yes (use PCA features)":
        # 動態計算 PCA
        pca = PCA(n_components=pca_n)
        X_model = pca.fit_transform(X_scaled)   # PCA 在 scaled 版本上計算
        feature_names = [f"PC{i+1}" for i in range(X_model.shape[1])]

        st.success(f"Using PCA with {pca_n} components. Shape = {X_model.shape}")

    else:
        X_model = X_scaled.copy()
        feature_names = X_numeric.columns
        st.info(f"Using original numeric features: shape = {X_model.shape}")

    # --- 確保 y 是一維 ---
    y_model = np.ravel(y)

    # --- Train / Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y_model, test_size=0.2, random_state=42
    )

    # --- 選擇模型 ---
    model_choice = st.selectbox(
        "Select Model",
        ["Linear Regression", "Random Forest Regressor"]
    )

    # --- 訓練模型 ---
    if model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42)
    
    model.fit(X_train, y_train)

    # --- 預測 ---
    y_pred = model.predict(X_test)

    # --- 顯示模型績效 ---
    st.subheader("Model Performance")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")

    # --- Actual vs Predicted & Residuals --- 
    st.subheader("Model Diagnostics")
    st.write("The plots below include four model diagnostic visuals: Actual vs Predicted, Residuals vs Fitted, the QQ Plot, and the Scale-Location Plot. Together, they help assess prediction accuracy, residual patterns, normality, and variance consistency.")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual WAR")
        ax.set_ylabel("Predicted WAR")
        ax.set_title(f"{model_choice}: Actual vs Predicted")
        st.pyplot(fig)

    with col2:
        residuals = y_test - y_pred

        fig, ax = plt.subplots(figsize=(6,5))
        ax.scatter(y_pred, residuals, alpha=0.6)
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

        ax.set_xlabel("Predicted WAR")
        ax.set_ylabel("Residual (Actual - Predicted)")
        ax.set_title(f"{model_choice}: Residual Plot (Predicted vs Residuals)")

        st.pyplot(fig)

    col7, col8 = st.columns(2)

    with col7:
        fig, ax = plt.subplots(figsize=(5,5))
        sm.qqplot(residuals, line='45', ax=ax)
        ax.set_title(f"{model_choice}: QQ Plot of Residuals")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        st.pyplot(fig)
    with col8:
        # Standardized residuals
        std_residuals = residuals / np.std(residuals)

        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(y_pred, np.sqrt(np.abs(std_residuals)), alpha=0.6)
        ax.set_xlabel("Predicted WAR")
        ax.set_ylabel("√|Standardized Residuals|")
        ax.set_title(f"{model_choice}: Scale-Location Plot")

        st.pyplot(fig)



    # --- PCA vs Original + Model 對比圖 (RMSE & R²) ---
    st.subheader("PCA vs Original Features: Model Comparison")
    st.write("This section compares model performance using original scaled features versus PCA-transformed features with basic methods (RMSE and R²).")
    results = []
    for feat_type, X_used in [("Original", X_scaled), ("PCA", X_pca)]:
        y_true = np.ravel(y)
        X_tr, X_te, y_tr, y_te = train_test_split(X_used, y_true, test_size=0.2, random_state=42)
        for model_name in ["Linear Regression", "Random Forest"]:
            if model_name == "Linear Regression":
                m = LinearRegression()
            else:
                m = RandomForestRegressor(n_estimators=300, random_state=42)
            m.fit(X_tr, y_tr)
            y_pred_temp = m.predict(X_te)
            results.append({
                "Feature": feat_type,
                "Model": model_name,
                "RMSE": np.sqrt(mean_squared_error(y_te, y_pred_temp)),
                "R2": r2_score(y_te, y_pred_temp)
            })

    results_df = pd.DataFrame(results)

    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(data=results_df, x="Model", y="RMSE", hue="Feature", ax=ax)
        ax.set_title("Model RMSE Comparison (Original vs PCA)")
        st.pyplot(fig)

    with col4:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(data=results_df, x="Model", y="R2", hue="Feature", ax=ax)
        ax.set_title("Model R² Comparison (Original vs PCA)")
        st.pyplot(fig)

    # --- 計算 Adjusted R² & AIC ---
    st.subheader("Adjusted R² & AIC Calculation")
    st.write("This section calculates Adjusted R² and AIC for models using both original and PCA features. These two methods provide effective ways to evaluate model performance while considering model complexity(adjusted R² to compare PCA/original features, and AIC to compare models).")
    n_test = y_te.shape[0]

    adjusted_results = []
    for feat_type, X_used in [("Original", X_scaled), ("PCA", X_pca)]:
        y_true = np.ravel(y)
        X_tr, X_te, y_tr, y_te = train_test_split(X_used, y_true, test_size=0.2, random_state=42)
        for model_name in ["Linear Regression", "Random Forest"]:
            if model_name == "Linear Regression":
                m = LinearRegression()
                m.fit(X_tr, y_tr)
                y_pred_temp = m.predict(X_te)
                
                # Adjusted R²
                r2_temp = r2_score(y_te, y_pred_temp)
                k = X_te.shape[1]  # 特徵數
                n = X_te.shape[0]  # 樣本數
                adj_r2 = 1 - (1 - r2_temp)*(n-1)/(n-k-1)
                
                # AIC (for linear regression)
                resid = y_te - y_pred_temp
                rss = np.sum(resid**2)
                aic = n * np.log(rss/n) + 2*k
                
            else:  # Random Forest
                m = RandomForestRegressor(n_estimators=300, random_state=42)
                m.fit(X_tr, y_tr)
                y_pred_temp = m.predict(X_te)
                
                # RF 沒有明確公式, 使用類似 R² 計算 Adjusted R²
                r2_temp = r2_score(y_te, y_pred_temp)
                k = X_te.shape[1]
                n = X_te.shape[0]
                adj_r2 = 1 - (1 - r2_temp)*(n-1)/(n-k-1)
                # AIC 用負對數似然近似
                resid = y_te - y_pred_temp
                rss = np.sum(resid**2)
                aic = n * np.log(rss/n) + 2*k

            adjusted_results.append({
                "Feature": feat_type,
                "Model": model_name,
                "Adjusted R2": adj_r2,
                "AIC": aic
            })

    adjusted_df = pd.DataFrame(adjusted_results)

    # --- 顯示 Adjusted R² & AIC 比較圖 ---
    st.subheader("Adjusted R² & AIC Comparison")

    col5, col6 = st.columns(2)

    with col5:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(data=adjusted_df, x="Model", y="Adjusted R2", hue="Feature", ax=ax)
        ax.set_title("Adjusted R² Comparison (Original vs PCA)")
        st.pyplot(fig)

    with col6:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(data=adjusted_df, x="Feature", y="AIC", hue="Model", ax=ax)
        ax.set_title("AIC Comparison by Feature Type (Original vs PCA)")
        st.pyplot(fig)


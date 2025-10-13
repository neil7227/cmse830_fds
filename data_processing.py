import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import tool

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

CPBL_data = pd.read_excel("CPBL_batter.xlsx")
MLB_data = pd.read_excel("MLB_batter.xlsx")

df = pd.concat([MLB_data, CPBL_data], axis=0, ignore_index=True)

Scale = 1.232
CPBL_Game = 120
MLB_Game = 162

CPBL_data.drop(columns=['BB/K','OPS','tOPS+','RC'], inplace=True)
MLB_data.drop(columns=['G','xwOBA','Def','SB'], inplace=True)
MLB_data['OPS+'] = 100 * (MLB_data['OBP']/MLB_data['OBP'].mean() + (MLB_data['SLG']/MLB_data['SLG'].mean()) - 1)
CPBL_data['Off'] = ((CPBL_data['wOBA'] / CPBL_data['wOBA'].mean()) - 1) * 100
CPBL_data.rename(columns={'球員': 'Name', '背號': 'Num', '隊伍': 'Team'}, inplace=True)

MLB_data[['K%', 'BB%']] = MLB_data[['K%', 'BB%']] * 100
CPBL_data['PA_scaled'] = CPBL_data['PA']/CPBL_Game
MLB_data['PA_scaled'] = MLB_data['PA']/MLB_Game

df = pd.concat([MLB_data, CPBL_data], axis=0, ignore_index=True)
df = df[df['PA_scaled'] > 2]

Scale = 600
df['HR_scaled'] = df['HR'] * Scale / df['PA']
df['R_scaled'] = df['R'] * Scale / df['PA']
df['RBI_scaled'] = df['RBI'] * Scale / df['PA']
df.drop(columns=['HR','R','RBI','PA'], inplace=True)

oe = OrdinalEncoder()
df['Num_Ordi'] = oe.fit_transform(df[['Num']])
df['Team_Ordi'] = oe.fit_transform(df[['Team']])
Num_original = np.array(sorted(df['Num'].unique()))

num_col = df.select_dtypes(include=[np.number]).columns
df_num = df[num_col].copy()
df_num.drop(columns=['Num','Num_Ordi','Team_Ordi'], inplace=True)

df_num["CPBL"] = False
df_num.loc[df_num["HR_scaled"].isnull() == True, "CPBL"] = True

num_cols = df_num.select_dtypes(include='number').columns
scaler = RobustScaler()
df_num[num_cols] = pd.DataFrame(scaler.fit_transform(df_num[num_cols]), columns=num_cols, index=df_num.index)

feature = ['ISO', 'SLG', 'OPS+', 'HR_scaled']
imputer = IterativeImputer(random_state=42, sample_posterior=True, max_iter=10)
df_num[feature] = imputer.fit_transform(df_num[feature])

df_clean = df_num.dropna(subset=['wRC+'])
X_train = df_clean[['Off', 'OPS+', 'wOBA']]
y_train = df_clean['wRC+']
linear_model = LinearRegression().fit(X_train, y_train)
missing = df_num.loc[df_num['wRC+'].isna(), ['Off', 'OPS+', 'wOBA']]
predicted_values = linear_model.predict(missing)
df_num.loc[df_num['wRC+'].isna(), 'wRC+'] = predicted_values

feature = ['ISO', 'SLG', 'RBI_scaled', 'HR_scaled']
df_num[feature] = imputer.fit_transform(df_num[feature])

feature = ['K%', 'BIP%', 'PutAway%', 'PA_scaled', 'AVG']
df_num[feature] = imputer.fit_transform(df_num[feature])

df_clean = df_num.dropna(subset=['R_scaled'])
X_train = df_clean[['Off', 'wRC+', 'wOBA']]
y_train = df_clean['R_scaled']
linear_model = LinearRegression().fit(X_train, y_train)
missing = df_num.loc[df_num['R_scaled'].isna(), ['Off', 'wRC+', 'wOBA']]
predicted_values = linear_model.predict(missing)
df_num.loc[df_num['R_scaled'].isna(), 'R_scaled'] = predicted_values

feature = ['PA_scaled','BB%','ISO','BABIP','AVG','OBP','SLG','wOBA','wRC+','Off','WAR','OPS+','HR_scaled','R_scaled']
df_num[feature] = imputer.fit_transform(df_num[feature])

feature_col = ['R_scaled','WAR','BsR']
rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1)
df_num[feature_col] = bsr_imputer.fit_transform(df_num[feature_col])

df_num['Num_Ordi'] = df['Num_Ordi']
imputer = KNNImputer(n_neighbors=3)
df_imputed = pd.DataFrame(imputer.fit_transform(df_num), columns=df_num.columns, index=df_num.index)

for col in df_num.columns:
    mask = df_num[col].isna()
    df_num.loc[mask, col] = df_imputed.loc[mask, col]

for col in df_num.columns:
    if col in df.columns:
        mask = df[col].isna()
        df.loc[mask, col] = df_num.loc[mask, col]

df_num.drop('BsR_missing', axis=1, inplace=True, errors='ignore')
df_num[num_cols] = pd.DataFrame(scaler.inverse_transform(df_num[num_cols]), columns=num_cols, index=df_num.index)
df_num = df_num[df_num['PA_scaled'] > 3]
df.update(df_num)
df['CPBL'] = df_num['CPBL']

df.to_csv("cleaned_data.csv", index=False)
df_num.to_csv("cleaned_data_num.csv", index=False)
print('finish')

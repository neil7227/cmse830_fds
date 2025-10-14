#!/usr/bin/env python
# coding: utf-8

# In[84]:


#import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import tool
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False 


# In[85]:


#read
CPBL_data = pd.read_excel("CPBL_batter.xlsx")
MLB_data = pd.read_excel("MLB_batter.xlsx")


# In[86]:


#first view
CPBL_data.info()
CPBL_data.describe()


# In[87]:


MLB_data.info()
MLB_data.describe()


# In[88]:


df = pd.concat([MLB_data, CPBL_data], axis=0, ignore_index=True)
df.info()


# In[89]:


#check column
same_col = set(CPBL_data.columns) & set(MLB_data.columns)
diff_col = set(CPBL_data.columns) ^ set(MLB_data.columns)
print("same column: ", same_col, "\ndiff. column: ", diff_col)


# In[90]:


#data clean
Scale = 1.232
CPBL_Game = 120
MLB_Game = 162

CPBL_data.drop(columns=['BB/K','OPS','tOPS+','RC'], inplace=True)
MLB_data.drop(columns=['G','xwOBA','Def','SB'], inplace=True)
MLB_data['OPS+'] = 100 * (MLB_data['OBP']/MLB_data['OBP'].mean() + (MLB_data['SLG']/MLB_data['SLG'].mean()) - 1)
CPBL_data['Off'] = CPBL_data['Off'] = ((CPBL_data['wOBA'] / CPBL_data['wOBA'].mean()) - 1) * 100
CPBL_data.rename(columns={
    '球員': 'Name',
    '背號': 'Num',
    '隊伍': 'Team'
}, inplace=True)
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


# In[91]:


#encoding
oe = OrdinalEncoder()
df['Num_Ordi'] = oe.fit_transform(df[['Num']])
df['Team_Ordi'] = oe.fit_transform(df[['Team']])
Num_original = np.array(sorted(df['Num'].unique()))
'''df['Num_Bi'] = df['Num_Ordi'].apply(tool.custom_binary_encode)
df['Team_Bi'] = df['Team_Ordi'].apply(tool.custom_binary_encode)'''
df.head()


# In[92]:


#numerical col
num_col = df.select_dtypes(include=[np.number]).columns
df_num = df[num_col]
df_num.drop(columns= ['Num','Num_Ordi','Team_Ordi'], inplace = True)
df_num.info()


# In[93]:


#split league
df_num["CPBL"] = False
df_num.loc[df_num["HR_scaled"].isnull() == True, "CPBL"] = True


# In[94]:


#standardize
num_cols = df_num.select_dtypes(include='number').columns
scaler = RobustScaler()

df_num[num_cols] = pd.DataFrame(
    scaler.fit_transform(df_num[num_cols]),
    columns=num_cols,
    index=df_num.index
)


# In[95]:


#heatmap
tool.plot_heatmap(df_num, 'full data heat map')


# In[96]:


#hist
col = [c for c in df_num.columns if c != 'CPBL']  

fig, axes = plt.subplots(5, 4, figsize=(40, 24))
axes = axes.flatten()

for i, c in enumerate(col):
    sns.histplot(data=df_num, x=c, hue='CPBL', bins=20, alpha=0.6, ax=axes[i])
    axes[i].set_title(c)

plt.tight_layout()
plt.show()
plt.tight_layout()
plt.show()


# In[97]:


'''df_clean = df_num.dropna(subset=['HR_scaled'])

X_train = df_clean[['ISO', 'SLG','OPS+']]
y_train = df_clean['HR_scaled']


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

missing = df_num.loc[df_num['HR_scaled'].isna(), ['ISO', 'SLG','OPS+']]
predicted_values = linear_model.predict(missing)

df_num.loc[df_num['HR_scaled'].isna(), 'HR_scaled'] = predicted_values
'''


# In[98]:


#impute
feature = ['ISO', 'SLG', 'OPS+', 'HR_scaled']

imputer = IterativeImputer(random_state=42, sample_posterior=True, max_iter=10)

imputed_array = imputer.fit_transform(df_num[feature])

df_num[feature] = imputed_array


# In[99]:


df_clean = df_num.dropna(subset=['wRC+'])

X_train = df_clean[['Off', 'OPS+', 'wOBA']]
y_train = df_clean['wRC+']


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

missing = df_num.loc[df_num['wRC+'].isna(), ['Off', 'OPS+', 'wOBA']]
predicted_values = linear_model.predict(missing)

df_num.loc[df_num['wRC+'].isna(), 'wRC+'] = predicted_values


# In[100]:


feature = ['ISO', 'SLG', 'RBI_scaled', 'HR_scaled']

imputer = IterativeImputer(random_state=42, sample_posterior=True, max_iter=10)

imputed_array = imputer.fit_transform(df_num[feature])

df_num[feature] = imputed_array


# In[101]:


feature = ['K%', 'BIP%', 'PutAway%', 'PA_scaled', 'AVG'] #後續train可加入['BB%', 'OBP', 'wOBA', 'Off', 'OPS+']看看效果 (tag)

imputer = IterativeImputer(random_state=42, sample_posterior=True, max_iter=10)

imputed_array = imputer.fit_transform(df_num[feature])

df_num[feature] = imputed_array


# In[102]:


df_clean = df_num.dropna(subset=['R_scaled'])

X_train = df_clean[['Off', 'wRC+', 'wOBA']]
y_train = df_clean['R_scaled']


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

missing = df_num.loc[df_num['R_scaled'].isna(), ['Off', 'wRC+', 'wOBA']]
predicted_values = linear_model.predict(missing)

df_num.loc[df_num['R_scaled'].isna(), 'R_scaled'] = predicted_values


# In[103]:


feature = ['PA_scaled', 'BB%', 'ISO', 'BABIP', 'AVG', 'OBP', 'SLG', 'wOBA', 'wRC+', 'Off', 'WAR', 'OPS+', 'HR_scaled', 'R_scaled']

imputer = IterativeImputer(random_state=42, sample_posterior=True, max_iter=10)

imputed_array = imputer.fit_transform(df_num[feature])

df_num[feature] = imputed_array


# In[104]:


feature_col = ['R_scaled', 'WAR', 'BsR'] #後續train可加入['BB%', 'BABIP', 'AVG', 'Off', 'HR_scaled', 'RBI_scaled']看看效果 (tag) 

rf = RandomForestRegressor(n_estimators=100, 
                           max_depth=10, 
                           min_samples_leaf=3,
                           random_state=42, 
                           n_jobs=-1
                           )

bsr_imputer = IterativeImputer(estimator=rf, random_state=42, max_iter=70, sample_posterior=False)
df_num[feature_col] = bsr_imputer.fit_transform(df_num[feature_col])


# In[105]:


df_num['Num_Ordi'] = df['Num_Ordi']
imputer = KNNImputer(n_neighbors=1)

df_imputed = pd.DataFrame(
    imputer.fit_transform(df_num),
    columns=df_num.columns,
    index=df_num.index
)

df_imputed.drop(columns=['PA_scaled'])

df_imputed.describe()
df.update(df_imputed)
df_num.update(df_imputed)
df['CPBL'] = df_imputed['CPBL']


# In[106]:


#tool.plot_heatmap(df_imputed, 'imputed data heat map')


# In[107]:


#show imputed
df_num.info()
df_num.drop('BsR_missing', axis=1, inplace=True, errors='ignore')
df_num.describe()


# In[108]:


df_num[num_cols] = pd.DataFrame(
    scaler.inverse_transform(df_num[num_cols]),
    columns=num_cols,
    index=df_num.index
)
df_num = df_num[df_num['PA_scaled'] >= 3]
df.update(df_num)
df = df[df['PA_scaled'] >= 3]


# In[109]:


#hist
col = [c for c in df_num.columns if c != 'CPBL']

fig, axes = plt.subplots(5, 4, figsize=(40, 24))
axes = axes.flatten()

for i, c in enumerate(col):
    sns.histplot(data=df_num, x=c, hue='CPBL', bins=20, alpha=0.6, ax=axes[i])
    axes[i].set_title(c)

plt.tight_layout()
plt.show()
plt.tight_layout()
plt.show()


# In[110]:


#update
df.update(df_num)
df['CPBL'] = df_num['CPBL'] 


# In[111]:


#scatter
col = [x for x in df.select_dtypes(include=np.number).columns 
       if x not in ['WAR']]


fig, axes = plt.subplots(5, 5, figsize=(40, 24))


for i, j in enumerate(col):
    sns.scatterplot(
    data=df,
    x=j,
    y='WAR',
    hue='CPBL',
    s=100,
    alpha=0.7,
    ax=axes[i//5, i%5]
    )
    axes[i//5, i%5].set_title(j, fontsize=16)
    axes[i//5, i%5].tick_params(labelsize=12)
    axes[i//5, i%5].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=2, fontsize=10)

plt.tight_layout()
plt.show()


# In[112]:


col = [x for x in df.select_dtypes(include=np.number).columns 
       if x not in ['OPS+', 'Num_Ordi', 'Team_Ordi']]


fig, axes = plt.subplots(4, 5, figsize=(40, 24))
palette = sns.color_palette("tab20", n_colors=df['Team'].nunique())


for i, j in enumerate(col):
    sns.scatterplot(
    data=df,
    x=j,
    y='OPS+',
    hue='Team',
    s=100,
    palette=palette,
    alpha=0.7,
    ax=axes[i//5, i%5]
    )
    axes[i//5, i%5].set_title(j, fontsize=16)
    axes[i//5, i%5].tick_params(labelsize=12)
    axes[i//5, i%5].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=2, fontsize=10)

plt.tight_layout()
plt.show()


# In[113]:


#violin 
plt.figure(figsize=(24,10))

sns.violinplot(
    data=df,
    x='Team',
    y='OPS+',
    palette='Set2', 
    inner='box',   
    width=0.9,
    hue='Team'
)

plt.xticks(rotation=90)

plt.xlabel('Team', fontsize=12)
plt.ylabel('OPS+', fontsize=12)
plt.title('OPS+/Team', fontsize=16)

plt.tight_layout()
plt.show()


# In[114]:


df.to_csv("cleaned_data.csv", index=False)
df_num.to_csv("cleaned_data_num.csv", index=False)


#import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

def plot_heatmap(dataframe, title): #define heatmap function
    plt.figure(figsize=(16, 12)) #define the plot size
    sns.heatmap(dataframe.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1) #same as heatmap above
    plt.title(title) #define title
    plt.show() #show

def custom_binary_encode(value):
    if pd.isna(value) or not np.isfinite(value):
        return None
    return format(int(value), '02b')
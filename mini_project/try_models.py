import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
import plt.pyplot as plt
import seaborn as sns




train_df2=pd.read_csv('')
for column in tqdm(columns_to_remove_outleirs):
    q1=train_df2.loc[:,column].quantile(0.25)
    q3=train_df2.loc[:, column].quantile(0.75)
    iqr=q3-q1
    upper=q3+1.5*iqr
    lower=q1-1.5*iqr
    train_df2=train_df2[train_df2.loc[:,column]<upper]
    train_df2=train_df2[train_df2.loc[:,column]>lower]

target=pd.get_dummies(train_df2.Credit_Score)
x_train,x_test, y_train, y_test = train_test_split(train_df2.iloc[:,1:-1],target,test_size=0.25,stratify=target, random_state=42)
pipeline1=make_pipeline(RobustScaler(),KNeighborsClassifier(n_neighbors=9,metric= 'manhattan', weights= 'distance'))
pipeline2=make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=7,metric= 'manhattan', weights= 'distance'))
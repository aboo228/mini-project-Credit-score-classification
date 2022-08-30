import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

path = r'train_df.csv'
df = pd.read_csv(path, low_memory=False)

# Payment_of_Min_Amount_get_dummies = pd.get_dummies(df['Payment_of_Min_Amount'], drop_first=True)
# df = pd.concat([df, Payment_of_Min_Amount_get_dummies], axis=1)
df = df.drop(['Unnamed: 0'], axis=1)

df = df.astype('float64')


df_to_train = df[~df['Amount_invested_monthly'].isnull()]
df_to_train = df_to_train[~[df_to_train['Amount_invested_monthly'] == 10000.00000][0]]

df_to_train = df_to_train[df_to_train.isnull().sum( axis = 1)==0]
describe = df_to_train.describe()


X, y = df_to_train.drop(['Amount_invested_monthly'], axis=1), df_to_train['Amount_invested_monthly']


transformer = RobustScaler().fit(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

reg = LinearRegression().fit(x_train, y_train)
reg.score(x_train, y_train)

reg.score(x_test, y_test)


reg.coef_

reg.intercept_

transformer.transform(X)

a = reg.predict(x_test[1:2])

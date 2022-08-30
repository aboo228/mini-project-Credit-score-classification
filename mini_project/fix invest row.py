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
import seaborn as sns

path = r'train_df.csv'
df = pd.read_csv(path, low_memory=False)

df['Amount_invested_monthly'] = df['Amount_invested_monthly'].replace('None', None)
df = df.astype('float64')

# Payment_of_Min_Amount_get_dummies = pd.get_dummies(df['Payment_of_Min_Amount'], drop_first=True)
# df = pd.concat([df, Payment_of_Min_Amount_get_dummies], axis=1)
df = df.drop(['Unnamed: 0'], axis=1)

df_to_train = df[~df['Amount_invested_monthly'].isnull()]
df_to_train = df_to_train[~[df_to_train['Amount_invested_monthly'] == 10000.00000][0]]
df_to_train = df_to_train[[df_to_train['Monthly_Balance'] < 2000][0]]
# df_to_train['Annual_Income'] = np.log(df_to_train['Annual_Income'])



df_to_train = df_to_train[df_to_train.isnull().sum( axis = 1)==0]
describe = df_to_train.describe()

X, y = df_to_train.drop(['Amount_invested_monthly'], axis=1), df_to_train['Amount_invested_monthly']

transformer = RobustScaler().fit(X)
X = transformer.transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

reg = LinearRegression().fit(x_train, y_train)
reg.score(x_train, y_train)

reg.score(x_test, y_test)


coef = reg.coef_

intercept = reg.intercept_



a = reg.predict(x_test[1:2])
print('end')


# sns.histplot(df.Annual_Income,log_scale=True)
# # plt.waitforbuttonpress()
# plt.show()
#
# sns.histplot(df_to_train.Monthly_Balance)
# # plt.waitforbuttonpress()
# plt.show()
#
# sns.boxplot(df_to_train.Monthly_Balance)
# # plt.waitforbuttonpress()
# plt.show()
# z = df['Annual_Income'].groupby(df['Annual_Income']).count()
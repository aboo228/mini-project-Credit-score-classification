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

# index_null_list = list(df.index[df['Annual_Income'] > 200000])
# for i in tqdm(index_null_list):
#     df_to_train.loc[i, 'Annual_Income'] = df['Annual_Income'][df['Customer_ID'] == df['Customer_ID'][i]].mode()[0]




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



print('end')
# prediction = []
# for i in tqdm(range(0, len(x_test))):
#     y_p = reg.predict(x_test[i:i+1])
#     prediction.append(y_p)
# loss = np.square(np.array(y_test) - np.array(prediction)).mean()**0.5

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
z = df['Annual_Income'].groupby(df['Annual_Income']).count()
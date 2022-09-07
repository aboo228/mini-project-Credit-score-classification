import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor,LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader
# from torchvision import datasets, transforms


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# import Credit_Score_Classification.py

path = r'train_df.csv'
train_df = pd.read_csv('train_df.csv')

# Payment_of_Min_Amount_get_dummies = pd.get_dummies(df['Payment_of_Min_Amount'], drop_first=True)
# df = pd.concat([df, Payment_of_Min_Amount_get_dummies], axis=1)


# df = df.astype('float64')
#
#
# df_to_train = df[~df['Amount_invested_monthly'].isnull()]
# df_to_train = df_to_train[~[df_to_train['Amount_invested_monthly'] == 10000.00000][0]]
instances_with_null = train_df.index[train_df.isnull().sum(axis=1) > 0]
columns_with_null = train_df.columns[train_df.isnull().sum() > 0]
instances_to_predict = train_df.iloc[instances_with_null, :]

df_to_train = train_df.drop(instances_with_null, axis=0)
'''drop Credit_History_Age'''

describe = df_to_train.describe()


class Model(nn.Model):
    def __init__(self,input_size,num_classes=3):
        super(NN,self).__init__()
        self.fc1=nn.Linear(input_size,50)
        self.fc2=nn.Linear(50,num_classes)

    def forward(self,x):
        x=nn.ReLU(self.fc1(x))
        x=self.fc2(x)
        return x




#
#
# for column in columns_with_null:
#     X, y = df_to_train.drop(column, axis=1), df_to_train[column]
#
#     transformer = RobustScaler().fit(X)
#     xs=X.copy()
#     xs=RobustScaler().fit(xs).transform(xs)
# # norm=StandardScaler().fit(X)
# #     X=transformer.transform(X)
#
# # X=norm.transform(X)
#     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     reg = LinearRegression()
#     reg.fit(x_train, y_train)
#     sgd=SGDRegressor(max_iter=100,eta0=0.001,n_iter_no_change=1000)
#     sgd.fit(x_train, y_train)
#     logistic=LogisticRegression()
#     # logistic.fit(x_train, y_train)
#     #
#
#     print('reg',reg.score(x_train, y_train))
#     # print('logistic',logistic.score(x_train, y_train))
#     print('sgd',sgd.score(x_train, y_train))
#
#     print('reg',reg.score(x_test, y_test))
#     # print('logistic',logistic.score(x_test, y_test))
#     print('sgd',sgd.score(x_test, y_test))
#
#     reg.coef_
#
#     reg.intercept_
#

# print(reg.predict(x_test[:15]))
# print(sgd.predict(x_test[:15]))
# print(reg.predict(x_train[:15]))
# print(sgd.predict(x_train[:15]))
    # xs=X.copy()
    # xs=RobustScaler().fit(xs).transform(xs)
# norm=StandardScaler().fit(X)
#     X=transformer.transform(X)

# X=norm.transform(X)
# X, y = df_to_train.drop(columns_with_null[0], axis=1), df_to_train[columns_with_null[0]]
#
# lr_=0.01
# X = RobustScaler().fit(X).transform(X)
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# nn_model=nn.Sequential(nn.Linear(x_train.shape[1],120),nn.ReLU(),nn.Linear(120,1),nn.Sigmoid())
# loss_=nn.MSELoss()
# optimizer= torch.optim.Adam(nn_model.parameters(),lr=lr_  )
# losses=[]
#
# for epoch in range(10):
#     pred=nn_model(torch.asarray(x_train)[0].reshape(-1,1))
#     loss=loss_(pred,torch.asarray(y_train)[epoch])
#     losses.append(loss.item)
#     nn_model.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#

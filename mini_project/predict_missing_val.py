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
from tqdm import trange, tqdm


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

# df_to_train.drop('Credit_Score',axis=1,inplace=True)


describe = df_to_train.describe()


class Model(nn.Module):
    def __init__(self,input_size,num_classes=3):
        super(Model,self).__init__()
        self.fc1=nn.Linear(input_size,50)
        self.fc2=nn.Linear(50,num_classes)

    def forward(self,x):
        x=nn.functional.relu(self.fc1(x))
        x=self.fc2(x)
        return x

model=Model(784,10)
x=torch.randn(64,784)
print(model(x).shape)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''Hyperparameters'''
input_size=df_to_train.shape[1]-1
num_classes=1
learning_rate=0.001
batch_size=1000
num_epochs=1
'''load data'''
df_to_train=pd.get_dummies(df_to_train)
df_to_train=df_to_train.astype(np.float64)
target=pd.DataFrame(df_to_train[columns_with_null[0]])
x_train,x_test,y_train,y_test=train_test_split(df_to_train.drop(columns_with_null[0],axis=1),target,random_state=42)
train_loader=DataLoader(dataset=tuple(zip(x_train.to_numpy(),y_train.to_numpy())),batch_size=batch_size,shuffle=True)
test_loader=DataLoader(dataset=tuple(zip(x_test.to_numpy(),y_test.to_numpy())),batch_size=batch_size,shuffle=True)
'''Initialize network'''
model=Model(input_size=input_size,num_classes=num_classes).to(device)
'''loss and optimizer'''
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

'''train network'''
for epoch in range(num_epochs):
    for batch_idc, (data,targets) in tqdm(enumerate(train_loader)):
        # get data to cuda
        data=data.to(device=device)
        targets=targets.to(device=device)

#         forward
        scores=model(data)
        loss=criterion(scores,targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

# check accuracy on traning & test to see hoe good our model
def check_accuracy(loader,model):
    num_correct=0
    num_samples=0
    model.eval()
    with torch.no_grad():
        for x,y in tqdm(loader):
            x=x.to(device=device)
            y=y.to(device=device)

            scores=model(x)
            _,predictions=scores.max(1)
            num_correct+=(predictions==y).sum()
            num_samples+=predictions.size(0)

        print(f'got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()
    return acc
check_accuracy(train_loader,model)
check_accuracy(test_loader,model)

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

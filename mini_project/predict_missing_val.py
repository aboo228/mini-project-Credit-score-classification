import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision
import optuna

# import Credit_Score_Classification.py

path = r'train_df.csv'
train_df2 = pd.read_csv('train_df.csv')
'''convert target values to numbers: poor:0 ,standard:1, good:2'''
# Payment_of_Min_Amount_get_dummies = pd.get_dummies(df['Payment_of_Min_Amount'], drop_first=True)
# df = pd.concat([df, Payment_of_Min_Amount_get_dummies], axis=1)
train_df2.drop('Customer_ID', axis=1, inplace=True)
convert_dict = {'Poor': 0, 'Standard': 1, 'Good': 2}
for (label, num) in convert_dict.items():
    train_df2.loc[train_df2.index[train_df2.loc[:, 'Credit_Score'] == label], 'Credit_Score'] = num

# df = df.astype('float64')
#
#
# df_to_train = df[~df['Amount_invested_monthly'].isnull()]
# df_to_train = df_to_train[~[df_to_train['Amount_invested_monthly'] == 10000.00000][0]]
instances_with_null = train_df2.index[train_df2.isnull().sum(axis=1) > 0]
columns_with_null = train_df2.columns[train_df2.isnull().sum() > 0]
instances_to_predict = train_df2.iloc[instances_with_null, :]

# df_to_train = train_df.drop(instances_with_null, axis=0)


train_df2.dropna(inplace=True)

columns_to_remove_outleirs = train_df2.columns[:17]
q1 = None
q3 = None
iqr = None
upper = None
lower = None

# for column in tqdm(columns_to_remove_outleirs):
#     q1 = train_df2.loc[:, column].quantile(0.25)
#     q3 = train_df2.loc[:, column].quantile(0.75)
#     iqr = q3 - q1
#     upper = q3 + 2 * iqr
#     lower = q1 - 2 * iqr
#     train_df2 = train_df2[train_df2.loc[:, column] < upper]
#     train_df2 = train_df2[train_df2.loc[:, column] > lower]

'''drop Credit_History_Age'''

# df_to_train.drop('Credit_Score',axis=1,inplace=True)


df_to_train=train_df2


# def objective()

# df_to_train=pd.get_dummies(df_to_train)
class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 60)
        # self.fc2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(60, 180)
        # self.fc4 = nn.Linear(50, 180)

        self.fc5 = nn.Linear(180, input_size)
        self.fc6 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x=nn.functional.leaky_relu(x)
        # x = self.fc2(x)
        x = self.fc3(x)
        # x = self.fc4(x)
        x = self.fc5(x)
        x = nn.functional.leaky_relu(x)
        # x=nn.functional.relu(x)
        x = self.fc6(x)
        return x


class Df(Dataset):
    def __init__(self):
        self.x = torch.tensor(x_train, dtype=torch.float32)
        self.y = torch.tensor(y_train.astype(np.int32), dtype=torch.int32).view(-1)
        self.instances = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.instances


df_to_train.iloc[:,:-1] = df_to_train.iloc[:,:-1].astype(np.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")
'''Hyperparameters'''
num_classes = 3
learning_rate = 0.0033
num_epochs = 2000

'''load data'''
target = pd.DataFrame(df_to_train['Credit_Score']).to_numpy()
# df_to_train=df_to_train.iloc[:,:17]
input_size = df_to_train.shape[1] - 1
# Y=pd.get_dummies(df_to_train['Credit_Score']).to_numpy()
x_train, x_test, y_train, y_test = train_test_split(df_to_train.drop('Credit_Score', axis=1), target, random_state=42,
                                                    stratify=target)
batch_size = round(x_train.shape[0])

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
df = Df()

train_loader = DataLoader(dataset=df, batch_size=batch_size, shuffle=False)
# test_loader=DataLoader(dataset=df,batch_size=batch_size,shuffle=False)
# train_loader=DataLoader(dataset=tuple(zip(x_train,y_train)),batch_size=batch_size,shuffle=False)
# test_loader=DataLoader(dataset=tuple(zip(x_test.to_numpy(),y_test.to_numpy())),batch_size=batch_size,shuffle=False)
'''Initialize network'''
model = Model(input_size=input_size, num_classes=num_classes).to(device)
'''loss and optimizer'''
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
losses = []
'''train network'''
if __name__ == '__main__':
    for epoch in tqdm(range(num_epochs)):
        for batch_idc, (data, targets) in enumerate(train_loader):
            # get data to cuda
            data = data.to(device=device)
            targets = targets.type(torch.LongTensor).to(device=device)

            #         forward
            predictions = model(data)
            loss = criterion(predictions, targets)

            # backward
            if epoch > 0:
                plt.scatter(epoch, loss.item())
                losses.append(loss.item())
                print(f'epoch:{epoch}\t,iter: {batch_idc}\t,loss:{loss.item()} ')
            if loss.item()<0.20:
                _=loss.item()
                torch.save(model.state_dict(),'best_loss2.pt')
                model_best_loss=Model(input_size=input_size, num_classes=num_classes)
                model_best_loss.load_state_dict(torch.load('best_loss'))
                break
            loss.backward()
                # gradient descent or adam step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    xx_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    xx_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        x_train_pred = model(xx_train).to('cpu')
        x_test_pred = model(xx_test).to('cpu')
    plt.show()

    acc_train = (torch.max(x_train_pred, 1)[1].numpy().reshape(-1, 1) == y_train).sum() / y_train.shape[0]
    acc_test = (torch.max(x_test_pred, 1)[1].numpy().reshape(-1, 1) == y_test).sum() / y_test.shape[0]
    print(f'train accuracy:{acc_train}\ntest accuracy:{acc_test}')

# img_grid=torchvision.utils.make_grid(torch.tensor(losses))
# writer.add_image('loss',img_grid)
# writer.close()


# check accuracy on traning & test to see hoe good our model
# def check_accuracy(loader,model):
#     num_correct=0
#     num_samples=0
#     model.eval()
#     with torch.no_grad():
#         for x,y in tqdm(loader):
#             x=x.to(device=device)
#             y=y.to(device=device)
#
#             scores=model(x)
#             _,predictions=scores.max(1)
#             num_correct+=(predictions==y).sum()
#             num_samples+=predictions.size(0)
#
#         print(f'got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
#
#     model.train()

# check_accuracy(train_loader,model)
# check_accuracy(test_loader,model)

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

#sdfsdf
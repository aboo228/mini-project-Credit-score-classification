import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,RandomForestClassifier,VotingClassifier
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC,LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns

train_df2 = pd.read_csv('train_df.csv').drop('Customer_ID', axis=1)
train_dfall = pd.read_csv('train_df.csv').drop('Customer_ID', axis=1)
train_dfall.dropna(inplace=True)
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
#     upper = q3 + 2.5 * iqr
#     lower = q1 - 2.5 * iqr
#     train_df2 = train_df2[train_df2.loc[:, column] < upper]
#     train_df2 = train_df2[train_df2.loc[:, column] > lower]

targetall_dum = pd.get_dummies(train_dfall.Credit_Score)
target = pd.get_dummies(train_df2.Credit_Score)
x_train, x_test, y_train, y_test = train_test_split(train_df2.iloc[:, :-1], target, test_size=0.25, stratify=target,
                                                    random_state=42)
pipe1 = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, metric='manhattan', weights='distance'))
pipe2= make_pipeline(RobustScaler(),AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),n_estimators=64*3) )
pipe3 = make_pipeline(RobustScaler(), GradientBoostingClassifier(n_estimators=670,learning_rate=0.1))
pipe4=make_pipeline(RobustScaler(), DecisionTreeClassifier(max_depth=12,max_features=19))
pipe5 = make_pipeline(RobustScaler(), RandomForestClassifier(max_depth=12, max_features=19))

acc_train = None
acc_test = None
acc_all = None
# klkl

convert_dict = {'Poor': 1, 'Standard': 2, 'Good': 0}
for (label, num) in convert_dict.items():
    train_dfall.loc[train_dfall.index[train_dfall.loc[:, 'Credit_Score'] == label], 'Credit_Score'] = num

train_dfall.Credit_Score=train_dfall.Credit_Score.astype('int64')
targetall = train_dfall.Credit_Score
train_predict={}
test_predict={}
all_predict={}
acc_test_dict=[]

pipe1.fit(x_train, y_train)
train_predict['pipe1']=np.argmax(pipe1.predict(x_train), axis=1)
test_predict['pipe1']=np.argmax(pipe1.predict(x_test), axis=1)
all_predict['pipe1']=np.argmax(pipe1.predict(train_dfall.iloc[:, :-1]), axis=1)
acc_train = (train_predict['pipe1'] == np.argmax(y_train.to_numpy(), axis=1)).sum()/y_train.shape[0]
acc_test = (test_predict['pipe1'] == np.argmax(y_test.to_numpy(), axis=1)).sum() / y_test.shape[0]
acc_test_dict.append(acc_test)
acc_all = (all_predict['pipe1'] == np.argmax(targetall_dum.to_numpy(),axis=1)).sum() / targetall.shape[0]
print(f'acc_train:{acc_train}\nacc_test:{acc_test}\nacc_all{acc_all}')

# target = train_df2.Credit_Score
y_train=np.argmax(y_train.to_numpy(),axis=1)
y_test=np.argmax(y_test.to_numpy(),axis=1)
'''import trained neural network'''

import torch
from predict_missing_val import model as deepmodel
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deepmodel.load_state_dict(torch.load('best_loss24.pt'))
deepmodel.parameters().close()
deepred_train=torch.argmax(deepmodel(torch.tensor(x_train.to_numpy(),dtype=torch.float32).to('cuda')),axis=1)
deepred_test=torch.argmax(deepmodel(torch.tensor(x_test.to_numpy(),dtype=torch.float32).to('cuda')),axis=1)
deepred_all=torch.argmax(deepmodel(torch.tensor(train_dfall.iloc[:, :-1].to_numpy(),dtype=torch.float32).to('cuda')),axis=1)
train_predict['NN']=deepred_train.to('cpu')
test_predict['NN']=deepred_test.to('cpu')
all_predict['NN']=deepred_all.to('cpu')

acc_test_dict.append((deepred_test==torch.tensor(y_test).long().to(device)).sum()/y_test.shape[0])


for i,pipe in tqdm(enumerate([pipe2, pipe3, pipe4, pipe5])):
    pipe.fit(x_train, y_train)
    train_predict[f'pipe_{+i}']=pipe.predict(x_train)
    test_predict[f'pipe_{+i}']=pipe.predict(x_test)
    all_predict[f'pipe_{+i}']=pipe.predict(train_dfall.iloc[:, :-1])
    acc_train = (train_predict[f'pipe_{+i}'] == y_train).sum()/y_train.shape[0]
    acc_test = (test_predict[f'pipe_{+i}'] == y_test).sum() / y_test.shape[0]
    acc_test_dict.append(acc_test)
    acc_all = (all_predict[f'pipe_{+i}'] == targetall.to_numpy()).sum() / targetall.shape[0]
    print(f'{str(pipe)}acc_train:{acc_train}\nacc_test:{acc_test}\nacc_all{acc_all}')


####################




# pipe3.fit(x_train, y_train)
# train_predict['pipe3']=pipe3.predict(x_train)
# test_predict['pipe3']=pipe3.predict(x_test)
# all_predict['pipe3']=pipe3.predict(train_dfall.iloc[:, :-1])
#
# acc_train = (train_predict['pipe3'] == y_train).sum() / y_train.shape[0]
# acc_test = (test_predict['pipe3'] == y_test).sum() / y_test.shape[0]
# acc_all = (all_predict['pipe3'] == targetall).sum() / targetall.shape[0]
# print(f'acc_train:{acc_train}\nacc_test:{acc_test}\nacc_all{acc_all}')
##################

from scipy import stats

targetall = targetall
voting_acc={}
def models_votes():
    vote_train=None
    vote_test=None
    vote_all=None
    for pipe_pred,data,targets,batch in zip([train_predict,test_predict,all_predict],\
                              [vote_train,vote_test,vote_all],[y_train,y_test,targetall],['train','test','all']):
        data=(pd.DataFrame(pipe_pred).mode(axis=1)).iloc[:,0].to_numpy()
        voting_acc[batch]=(data==targets).sum()/targets.shape[0]
        print(voting_acc[batch])

# pip1 = np.argmax(pipeline1.predict(x), axis=1)
# pip2 = np.argmax(pipeline2.predict(x), axis=1)
# pip3 = np.argmax(pipeline3.predict(x), axis=1)
# votes = np.max(stats.mode(np.array([pip1, pip2, pip3]))[0])
models_votes()
#
# class Model(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(Model, self).__init__()
#         self.fc1 = nn.Linear(input_size, 50)
#         # self.fc2 = nn.Dropout(0.5)
#         self.fc3 = nn.Linear(50, 180)
#         # self.fc4 = nn.Dropout(0.4)
#         self.fc5 = nn.Linear(180, input_size)
#         self.fc6 = nn.Linear(input_size, num_classes)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x=nn.functional.leaky_relu(x)
#         # x = self.fc2(x)
#         x = self.fc3(x)
#         # x = self.fc4(x)
#         x = self.fc5(x)
#         x = torch.sigmoid(x)
#         x = self.fc6(x)
#         return x
#

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

train_df2 = pd.read_csv('train_df.csv').drop('Customer_ID', axis=1)
train_dfall = pd.read_csv('train_df.csv').drop('Customer_ID', axis=1)
train_dfall.dropna(inplace=True)
train_df2.dropna(inplace=True)

columns_to_remove_outleirs = train_df2.columns[1:17]
q1 = None
q3 = None
iqr = None
upper = None
lower = None

for column in tqdm(columns_to_remove_outleirs):
    q1 = train_df2.loc[:, column].quantile(0.25)
    q3 = train_df2.loc[:, column].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 3 * iqr
    lower = q1 - 3 * iqr
    train_df2 = train_df2[train_df2.loc[:, column] < upper]
    train_df2 = train_df2[train_df2.loc[:, column] > lower]

targetall = pd.get_dummies(train_dfall.Credit_Score)
target = pd.get_dummies(train_df2.Credit_Score)
x_train, x_test, y_train, y_test = train_test_split(train_df2.iloc[:, 1:-1], target, test_size=0.25, stratify=target,
                                                    random_state=42)
pipe1 = make_pipeline(RobustScaler(), KNeighborsClassifier(n_neighbors=9, metric='manhattan', weights='distance'))
pipe2 = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, metric='manhattan', weights='distance'))
pipe3 = make_pipeline(RobustScaler(), GradientBoostingClassifier(n_estimators=670,learning_rate=0.1))

acc_train = None
acc_test = None
acc_all = None
# klkl


train_predict={}
test_predict={}
all_predict={}

for i,pipe in enumerate(tqdm([pipe1, pipe2])):
    pipe.fit(x_train, y_train)
    train_predict[f'pipe_{+i}']=np.argmax(pipe.predict(x_train), axis=1)
    test_predict[f'pipe_{+i}']=np.argmax(pipe.predict(x_test), axis=1)
    all_predict[f'pipe_{+i}']=np.argmax(pipe.predict(train_dfall.iloc[:, 1:-1]), axis=1)
    acc_train = (train_predict[f'pipe_{+i}'] == np.argmax(y_train.to_numpy(), axis=1)).sum()/y_train.shape[0]
    acc_test = (test_predict[f'pipe_{+i}'] == np.argmax(y_test.to_numpy(), axis=1)).sum() / y_test.shape[0]
    acc_all = (all_predict[f'pipe_{+i}'] == np.argmax(targetall.to_numpy(),axis=1)).sum() / targetall.shape[0]
    print(f'acc_train:{acc_train}\nacc_test:{acc_test}\nacc_all{acc_all}')


####################


convert_dict = {'Poor': 1, 'Standard': 2, 'Good': 0}
for (label, num) in convert_dict.items():
    train_dfall.loc[train_dfall.index[train_dfall.loc[:, 'Credit_Score'] == label], 'Credit_Score'] = num

train_dfall.Credit_Score=train_dfall.Credit_Score.astype('int64')
targetall = train_dfall.Credit_Score
# target = train_df2.Credit_Score
y_train=np.argmax(y_train.to_numpy(),axis=1)
y_test=np.argmax(y_test.to_numpy(),axis=1)

# x_train, x_test, y_train, y_test = train_test_split(train_df2.iloc[:, 1:-1], target, test_size=0.25, stratify=target,random_state=42)

pipe3.fit(x_train, y_train)
train_predict['pipe3']=pipe3.predict(x_train)
test_predict['pipe3']=pipe3.predict(x_test)
all_predict['pipe3']=pipe3.predict(train_dfall.iloc[:, 1:-1])

acc_train = (train_predict['pipe3'] == y_train).sum() / y_train.shape[0]
acc_test = (test_predict['pipe3'] == y_test).sum() / y_test.shape[0]
acc_all = (all_predict['pipe3'] == targetall.to_numpy()).sum() / targetall.shape[0]
print(f'acc_train:{acc_train}\nacc_test:{acc_test}\nacc_all{acc_all}')
##################

from scipy import stats

targetall = targetall.to_numpy()
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

pip1 = np.argmax(pipeline1.predict(x), axis=1)
pip2 = np.argmax(pipeline2.predict(x), axis=1)
pip3 = np.argmax(pipeline3.predict(x), axis=1)
votes = np.max(stats.mode(np.array([pip1, pip2, pip3]))[0])
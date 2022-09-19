import pandas as pd
import numpy as np
from tqdm import tqdm
import sklearn.tree
from sklearn.model_selection import train_test_split,learning_curve,cross_val_score
from sklearn.linear_model import LinearRegression, SGDRegressor,Ridge,ElasticNet
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor, AdaBoostRegressor
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors,
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import optuna

# def objective(trial):


train=pd.read_csv("train_df.csv")
train.drop('Customer_ID',axis=1,inplace=True)
convert_dict={'Poor':0,'Standard':1,'Good':2}
for (label,num) in convert_dict.items():
    train.loc[train.index[train.loc[:,'Credit_Score']==label],'Credit_Score']=num

# annual_income_loutliers=train.index[train.loc[:,'Annual_Income']>170000]


# train=train.drop(annual_income_loutliers,axis=0)

instances_with_null = train.index[train.isnull().sum(axis=1) > 0]
columns_with_null = train.columns[train.isnull().sum() > 0]
instances_to_predict = train.loc[instances_with_null, :]
# train=pd.concat([train, train],axis=1)
train=train.drop(instances_with_null,axis=0)
# train=train.iloc[:,:17]
# train=train.iloc[:,1:]
test_pred=None
train_pred=None

# for column in columns_with_null:
#     train_na=train.drop(column,axis=1)
#     x_train,x_test,y_train,y_test= train_test_split(train.drop(column,axis=1),train.loc[:,column],test_size=0.2,random_state=42)
#
#     def objective(trial):
#         estimator=trial.suggest_categorical('estimator', ['sgdregression','linearregression','lasso'])
#         if estimator=='sgdregression':
#             penalty=trial.suggest_categorical('penalty', ['l2', 'l1'])
#             loss=trial.suggest_categorical('loss',['huber','squared_error'])
#             estimator=SGDRegressor(eta0=0.001,penalty=penalty,loss=loss,max_iter=10000)
#
#         else:
#             estimator=LinearRegression()
#
#         pipline=make_pipeline(RobustScaler(),estimator)
#         score=cross_val_score(pipline,x_train,y_train)
#         return score.mean()
#
    #
    # study1=optuna.create_study(direction='maximize')
    # study1.optimize(objective, n_trials=10)
    # print(study1.best_trial)

for column in tqdm(columns_with_null):
    train_na=train.drop(column,axis=1)
    x_train,x_test,y_train,y_test= train_test_split(train.drop(column,axis=1),train.loc[:,column],test_size=0.25,random_state=42)
    model=RandomForestRegressor(n_estimators=35)
    # model=GradientBoostingRegressor(learning_rate=0.1,n_estimators=70)
    # model=LinearRegression()
    # model=ElasticNet(max_iter=1000)
    # model=SVR()
    # model=AdaBoostRegressor(n_estimators=320)
    # if True:
    #     x_train = x_train.astype(np.float32)
    #     x_train[x_train > 0] = np.log(x_train[x_train > 0])
    #     x_test = x_test.astype(np.float32)
    #     x_test[x_test > 0] = np.log(x_test[x_test > 0])
    # scaler=StandardScaler()
    pipline=make_pipeline(model)
    pipline.fit(x_train,y_train)
    train_pred=pipline.predict(x_train)
    test_pred=pipline.predict(x_test)
    print(f'test score: {pipline.score(x_test,y_test)}')
    print(f'train score:{pipline.score(x_train,y_train)}')
    print('train mean residual values:',(np.abs(train_pred - y_train)).sum() / y_train.shape[0])
    print('test mean residual values:',(np.abs(test_pred - y_test)).sum() / y_train.shape[0])
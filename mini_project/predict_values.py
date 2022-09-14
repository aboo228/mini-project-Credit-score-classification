import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,RobustScaler
train=pd.read_csv("train_df.csv")
convert_dict={'Poor':0,'Standard':1,'Good':2}
for (label,num) in convert_dict.items():
    train.loc[train.index[train.loc[:,'Credit_Score']==label],'Credit_Score']=num

instances_with_null = train.index[train.isnull().sum(axis=1) > 0]
columns_with_null = train.columns[train.isnull().sum() > 0]
instances_to_predict = train.iloc[instances_with_null, :]
# train=pd.concat([train, train],axis=1)
train=train.drop(instances_with_null,axis=0)
test_pred=None
train_pred=None

for column in columns_with_null:
    train_na=train.drop(column,axis=1)
    x_train,x_test,y_train,y_test= train_test_split(train.drop(column,axis=1),train.loc[:,column],test_size=0.2,random_state=42)
    model=LinearRegression()
    scaler=RobustScaler()
    pipline=make_pipeline(scaler,model)
    pipline.fit(x_test,y_test)
    train_pred=pipline.predict(x_train)
    test_pred=pipline.predict(x_test)
    print(f'test score: {pipline.score(x_test,y_test)}')
    print(f'train score:{pipline.score(x_train,y_train)}')
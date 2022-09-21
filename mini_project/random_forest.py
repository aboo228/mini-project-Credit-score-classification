import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

path = r'train_df.csv'
train_df = pd.read_csv('train_df.csv')
train_df.drop('Customer_ID',axis=1, inplace=True)

convert_dict = {'Poor': 0, 'Standard': 1, 'Good': 2}
for (label, num) in convert_dict.items():
    train_df.loc[train_df.index[train_df.loc[:, 'Credit_Score'] == label], 'Credit_Score'] = num

train_df.iloc[:, :-1] = train_df.iloc[:, :-1].astype('float32')
train_df.iloc[:, 1] = train_df.iloc[:, 1].astype('int32')

instances_with_null = train_df.index[train_df.isnull().sum(axis=1) > 0]
columns_with_null = train_df.columns[train_df.isnull().sum() > 0]
instances_to_predict = train_df.iloc[instances_with_null, :]

df_to_train = train_df.drop(instances_with_null, axis=0)

columns_to_remove_outleirs = df_to_train.columns[1:17]
q1 = None
q3 = None
iqr = None
upper = None
lower = None

for column in tqdm(columns_to_remove_outleirs):
    q1 = df_to_train.loc[:, column].quantile(0.25)
    q3 = df_to_train.loc[:, column].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 3 * iqr
    lower = q1 - 3 * iqr
    df_to_train = df_to_train[df_to_train.loc[:, column] < upper]
    df_to_train = df_to_train[df_to_train.loc[:, column] > lower]

x_train, x_test, y_train, y_test = train_test_split(df_to_train.iloc[:, :17], df_to_train.iloc[:, -1],stratify=df_to_train.iloc[:, -1])
y_train=y_train.to_numpy().astype(np.float32)
x_train=x_train.to_numpy().astype(np.int32)

def objective(trial):
    max_depth=trial.suggest_int('max_depth',3,5)
    n_estimators = trial.suggest_int('n_estimators',57,401)

    estimator=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
    # scalers=trial.suggest_categorical('scalers',['standard','robust'])
    #
    # if scalers == 'standard':
    #     scaler = StandardScaler()
    # else:
    #     scaler = RobustScaler()


    pipeline=make_pipeline(estimator)
    return cross_val_score(pipeline,x_train,y_train,cv=5).mean()

tree_study=optuna.create_study(direction='maximize')
tree_study.optimize(objective, n_trials=10)





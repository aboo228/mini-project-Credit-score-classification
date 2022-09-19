import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split
# from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score

path = r'train_df.csv'
train_df = pd.read_csv('train_df.csv')
train_df.drop('Customer_ID',axis=1,inplace=True)
train_df.drop('Num_of_Delayed_Payment',axis=1,inplace=True)

convert_dict = {'Poor': 0, 'Standard': 1, 'Good': 2}
for (label, num) in convert_dict.items():
    train_df.loc[train_df.index[train_df.loc[:, 'Credit_Score'] == label], 'Credit_Score'] = num

train_df.iloc[:, :-1] = train_df.iloc[:, :-1].astype('float32')
train_df.iloc[:, 1] = train_df.iloc[:, 1].astype('int32')

instances_with_null = train_df.index[train_df.isnull().sum(axis=1) > 0]
columns_with_null = train_df.columns[train_df.isnull().sum() > 0]
instances_to_predict = train_df.iloc[instances_with_null, :]

df_to_train = train_df.drop(instances_with_null, axis=0)

x_train, x_test, y_train, y_test = train_test_split(df_to_train.iloc[:, :17], df_to_train.iloc[:, -1],stratify=df_to_train.iloc[:, -1])
y_train=y_train.to_numpy().astype(np.float32)
x_train=x_train.to_numpy().astype(np.int32)

def objective(trial):
    # scaling metrics to choose from
    scalers = trial.suggest_categorical('scalers', ['standard', 'robust'])

    # define scalers
    if scalers == 'standard':
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    # dim_reduc = trial.suggest_categorical('dim_reduc', ['PCA', 'SVD', None])
    #
    # if dim_reduc == 'PCA':
    #     pca_n_components = trial.suggest_int('pca_n_components', 4, 30)
    #     dim_reduc_algo = PCA(n_components=pca_n_components)
    #
    # elif dim_reduc == 'SVD':
    #
    #     svd_n_components = trial.suggest_int('svd_n_components', 4, 30)
    #     dim_reduc_algo = TruncatedSVD(n_components=svd_n_components)
    # else:
    #     dim_reduc_algo = 'passthrough'
    # # integer from 1 to 19 with steps of 2
    knn_n_neighbors = trial.suggest_int('knn_n_neighbors', 7, 15, 2)
    knn_metric = trial.suggest_categorical('knn_metric', ['manhattan'])
    knn_weights = trial.suggest_categorical('knn_weights', ['uniform', 'distance'])

    estimator = KNeighborsClassifier(n_neighbors=knn_n_neighbors, metric=knn_metric, weights=knn_weights)

    # make_pipeline
    pipeline = make_pipeline(scaler, estimator)

    # evaluate score by cross_validation
    score = cross_val_score(pipeline, x_train, y_train,cv=10)
    # f1 = score()
    return score.mean()



study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
print(study.best_trial)

def objectivetree(trial):
    max_depth=int(trial.suggest_discrete_uniform('max_depth',1,5,1))
    n_estimators = trial.suggest_int('n_estimators',10,700)
    model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
    return cross_val_score(model,x_train,y_train,cv=10).mean()

tree_study=optuna.create_study(direction='maximize')
tree_study.optimize(objectivetree, n_trials=10)
print(tree_study.best_trial)


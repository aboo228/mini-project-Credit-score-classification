import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
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
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr
    train_df2 = train_df2[train_df2.loc[:, column] < upper]
    train_df2 = train_df2[train_df2.loc[:, column] > lower]

targetall = pd.get_dummies(train_dfall.Credit_Score)
target = pd.get_dummies(train_df2.Credit_Score)
x_train, x_test, y_train, y_test = train_test_split(train_df2.iloc[:, 1:-1], target, test_size=0.25, stratify=target,
                                                    random_state=42)
pipeline1 = make_pipeline(RobustScaler(), KNeighborsClassifier(n_neighbors=9, metric='manhattan', weights='distance'))
pipeline2 = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, metric='manhattan', weights='distance'))
pipeline3 = make_pipeline(RobustScaler(), GradientBoostingClassifier())
acc_train = None
acc_test = None
acc_all = None

for pipe in tqdm([pipeline1, pipeline2]):
    pipe.fit(x_train, y_train)
    acc_train = (np.argmax(pipe.predict(x_train), axis=1) == np.argmax(y_train.to_numpy(), axis=1)).sum()/y_train.shape[0]
    acc_test = (np.argmax(pipe.predict(x_test), axis=1) == np.argmax(y_test.to_numpy(), axis=1)).sum() / y_test.shape[0]
    acc_all = (np.argmax(pipe.predict(train_dfall.iloc[:, 1:-1]), axis=1) == np.argmax(targetall.to_numpy(),axis=1)).sum() / targetall.shape[0]
    print(f'acc_train:{acc_train}\nacc_test:{acc_test}\nacc_all{acc_all}')


##################
convert_dict = {'Poor': 0, 'Standard': 1, 'Good': 2}
data_list = [train_df2, train_dfall]
for data in data_list:
    for (label, num) in convert_dict.items():
        data.loc[data.index[data.loc[:, 'Credit_Score'] == label], 'Credit_Score'] = float(num)
    data['Credit_Score'] = data['Credit_Score'].astype('float32')

targetall = train_dfall.Credit_Score
target = train_df2.Credit_Score
x_train, x_test, y_train, y_test = train_test_split(train_df2.iloc[:, 1:-1], target, test_size=0.25, stratify=target,random_state=42)

pipeline3.fit(x_train, y_train)
acc_train = (pipeline3.predict(x_train) == y_train.to_numpy()).sum() / y_train.shape[0]
acc_test = (pipeline3.predict(x_test) == y_test.to_numpy()).sum() / y_test.shape[0]
acc_all = (pipeline3.predict(train_dfall.iloc[:, 1:-1]) == targetall.to_numpy()).sum() / targetall.shape[0]
print(f'acc_train:{acc_train}\nacc_test:{acc_test}\nacc_all{acc_all}')
##################

from scipy import stats


def models_votes(x, target):
    pip1 = np.argmax(pipeline1.predict(x), axis=1)
    pip2 = np.argmax(pipeline2.predict(x), axis=1)
    pip3 = np.argmax(pipeline3.predict(x), axis=1)
    votes = np.max(stats.mode(np.array([pip1, pip2, pip3]))[0])

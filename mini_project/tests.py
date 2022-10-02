import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from tqdm import trange, tqdm
train_path = r'train_df.csv'

train_df = pd.read_csv(train_path)
train_df = train_df.dropna()
train_df = train_df.drop(['Customer_ID'], axis=1)

convert_dict = {'Poor': 0, 'Standard': 1, 'Good': 2}
for (label, num) in convert_dict.items():
    train_df.loc[train_df.index[train_df.loc[:, 'Credit_Score'] == label], 'Credit_Score'] = float(num)

train_df['Credit_Score'] = train_df['Credit_Score'].astype('float32')

X = train_df.drop(['Credit_Score'], axis=1)
y = train_df['Credit_Score']
# y = pd.get_dummies(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

# percent of freqec target
(y.groupby(y).count() / len(y)).round(2)

# def predict_val(data, y):
#     prediction = []
#     for i in tqdm(range(0, len(X_val))):
#         y_p = clf.best_estimator_.predict(X[i:i+1])
#         prediction.append(y_p[0])
#     # print(f'{prediction}\n{list(y_val)}')
#
#     print(f'accuracy val {(prediction == y_val).sum()/len(X_val)} accuracy train {clf.best_score_}')

print('start to run test')


# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import RandomForestClassifier
#
# random_forest = RandomForestClassifier(random_state=42)
#
#
# adaboost = AdaBoostClassifier(random_state=0)
# parameters = {'base_estimator': [random_forest], 'n_estimators': [5, 10, 15, 20, 25, 30], 'learning_rate':[0.5, 1, 1.5]}
# clf = GridSearchCV(adaboost, parameters, scoring='accuracy')
# clf.fit(X_train, y_train)
# print(f'adaboost:  best_score {clf.best_score_} best_params {clf.best_params_}')

from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=9)

neigh.fit(X_train)
# print(f'neigh:  best_score {clf.best_score_} best_params {clf.best_params_}')

# neigh.score(X_val, y_val)

# predict_val(X_val, y_val)
predict(X_val)

print('finish to run test')

print(neigh.kneighbors(X_val))

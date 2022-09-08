import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
train_path = r'train_df.csv'

train_df = pd.read_csv(train_path)
train_df = train_df.dropna()
train_df = train_df.drop(['Unnamed: 0'], axis=1)


X = train_df.drop(['Credit_Score_Standard'], axis=1)
y = train_df['Credit_Score_Standard']

pca = PCA(n_components=54)
pca.fit(X)
X = pca.fit_transform(X)

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

random_forest = RandomForestClassifier(random_state=42)
# print(cross_val_score(random_forest, X, y))
parameters = {'max_depth': [1, 2]}
clf = GridSearchCV(random_forest, parameters, scoring='accuracy')
clf.fit(x_train, y_train)
print(f'best_score {clf.best_score_} best_params {clf.best_params_}')


def predict_test(num):
    prediction = []
    for i in tqdm(range(0, num)):
        y_p = clf.best_estimator_.predict(X[i:i+1])
        prediction.append(y_p[0])
    print(f'{prediction}\n{list(y[:num])}')

    return prediction

def predict_val(data, y):
    prediction = []
    for i in tqdm(range(0, len(data))):
        y_p = clf.best_estimator_.predict(X[i:i+1])
        prediction.append(y_p[0])
    # print(f'{prediction}\n{list(y_val)}')

    print(f'accuracy val {(prediction == y_val).sum()/len(x_val)} accuracy train {clf.best_score_}')

    return prediction


predict_val(x_val, y_val)

# test = predict_test(20)


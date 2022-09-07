import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV


train_path = r'train_df.csv'

train_df = pd.read_csv(train_path)
train_df = train_df.dropna()

X = train_df.drop(['Credit_Score_Standard'], axis=1)
y = train_df['Credit_Score_Standard']
random_forest = RandomForestClassifier(random_state=0)
# print(cross_val_score(random_forest, X, y))
parameters = {'max_depth': [1, 4]}
clf = GridSearchCV(random_forest, parameters, scoring='accuracy')
clf.fit(X, y)
print(f'best_score {clf.best_score_} best_params {clf.best_params_}')
# clf.best_estimator_.predict(X[4:5])
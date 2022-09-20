import pandas as pd
import numpy as np
# import seaborn as sns
from tqdm import trange, tqdm
# import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier


from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC



train_path = r'train_df.csv'

train_df = pd.read_csv(train_path)
train_df = train_df.dropna()
train_df = train_df.drop(['Customer_ID'], axis=1)

#todo train_df without get_dummies columns
# train_df['Credit_Score']= pd.get_dummies(train_df['Credit_Score'])['Good']

# convert_dict={'Poor':0,'Standard':1,'Good':2}
# for (label,num) in convert_dict.items():
#     train_df.loc[train_df.index[train_df.loc[:,'Credit_Score']==label],'Credit_Score']=float(num)


# train_df['Credit_Score'] = train_df['Credit_Score'].astype('float32')
#todo Credit_Score - get_dummies good no good

X = train_df.drop(['Credit_Score'], axis=1)
y = train_df['Credit_Score']
# y = pd.get_dummies(y)

def predict_val(data, y):
    prediction = []
    for i in tqdm(range(0, len(X_val))):
        y_p = clf.best_estimator_.predict(X[i:i+1])
        prediction.append(y_p[0])
    # print(f'{prediction}\n{list(y_val)}')

    print(f'accuracy val {(prediction == y_val).sum()/len(X_val)} accuracy train {clf.best_score_}')

# pca = PCA(n_components=54)
# pca.fit(X)
# X = pca.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

'''check random forest model'''
random_forest = RandomForestClassifier(random_state=42)

parameters = {'max_depth': [2, 4, 6, 8], 'n_estimators': [20, 30, 40, 50, 60]}
clf = GridSearchCV(random_forest, parameters, scoring='accuracy')
clf.fit(X_train, y_train)
print(f'random_forest:  best_score {clf.best_score_} best_params {clf.best_params_}')
predict_val(X_val, y_val)

'''check  Gradient Boosting model'''
Gradient_Boosting = GradientBoostingClassifier(learning_rate=1.0, random_state=0)

parameters = {'max_depth': [1, 2, 3, 4], 'n_estimators': [20, 30, 40, 50, 60], 'learning_rate':[0.1, 1]}
clf = GridSearchCV(Gradient_Boosting, parameters, scoring='accuracy')
clf.fit(X_train, y_train)
print(f'Gradient_Boosting:  best_score {clf.best_score_} best_params {clf.best_params_}')
predict_val(X_val, y_val)

'''check adaboost'''
adaboost = AdaBoostClassifier(random_state=0)
parameters = {'base_estimator': [random_forest], 'n_estimators': [20, 30, 40, 50, 60], 'learning_rate':[0.1, 1]}
clf = GridSearchCV(adaboost, parameters, scoring='accuracy')
clf.fit(X_train, y_train)
print(f'adaboost:  best_score {clf.best_score_} best_params {clf.best_params_}')
predict_val(X_val, y_val)

'''check  SVM model'''
# svm = svm.SVC(kernel='rbf')
#
# parameters = {'C': [1], 'degree': [3]}
# clf = GridSearchCV(svm, parameters, scoring='accuracy')
# clf.fit(X_train, y_train)
# print(f'best_score {clf.best_score_} best_params {clf.best_params_}')


#
# clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# clf.fit(X, y)


# def predict_test(num):
#     prediction = []
#     for i in tqdm(range(0, num)):
#         y_p = clf.best_estimator_.predict(X[i:i+1])
#         prediction.append(y_p[0])
#     print(f'{prediction}\n{list(y[:num])}')
#
#     return prediction
#


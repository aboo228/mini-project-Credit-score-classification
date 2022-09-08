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

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tensorflow import keras
from tensorflow.keras import layers
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense


train_path = r'train_df.csv'
#todo train_df without get_dummies columns
#todo Credit_Score - get_dummies good no good
train_df = pd.read_csv(train_path)
train_df = train_df.dropna()


X = train_df.drop(['Credit_Score'], axis=1)
y = train_df['Credit_Score']

# pca = PCA(n_components=54)
# pca.fit(X)
# X = pca.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=True)

'''check random forest model'''
# random_forest = RandomForestClassifier(random_state=42, n_estimators=50)
#
# parameters = {'max_depth': [10, 8]}
# clf = GridSearchCV(random_forest, parameters, scoring='accuracy')
# clf.fit(X_train, y_train)
# print(f'best_score {clf.best_score_} best_params {clf.best_params_}')

'''check  Gradient Boosting model'''
# Gradient_Boosting = GradientBoostingClassifier(learning_rate=1.0, random_state=0')
#
# parameters = {'max_depth': [0, 1], 'n_estimators': [99, 100]}
# clf = GridSearchCV(Gradient_Boosting, parameters, scoring='accuracy')
# clf.fit(X_train, y_train)
# print(f'best_score {clf.best_score_} best_params {clf.best_params_}')
#
# '''check  SVM model'''
# svm = svm.SVC(kernel='rbf')
#
# parameters = {'C': [1], 'degree': [3]}
# clf = GridSearchCV(svm, parameters, scoring='accuracy')
# clf.fit(X_train, y_train)
# print(f'best_score {clf.best_score_} best_params {clf.best_params_}')


#
# clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# clf.fit(X, y)

# #todo NN wite keras
#
# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(56,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=20, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))

# def predict_test(num):
#     prediction = []
#     for i in tqdm(range(0, num)):
#         y_p = clf.best_estimator_.predict(X[i:i+1])
#         prediction.append(y_p[0])
#     print(f'{prediction}\n{list(y[:num])}')
#
#     return prediction
#
# def predict_val(data, y):
#     prediction = []
#     for i in tqdm(range(0, len(X_val))):
#         y_p = clf.best_estimator_.predict(X[i:i+1])
#         prediction.append(y_p[0])
#     # print(f'{prediction}\n{list(y_val)}')
#
#     print(f'accuracy val {(prediction == y_val).sum()/len(X_val)} accuracy train {clf.best_score_}')
#
#     return prediction
#
# # test = predict_test(20)
#
# predict_val(X_val, y_val)
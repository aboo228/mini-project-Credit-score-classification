
import tensorflow as tf
from tensorflow import keras
from keras import layers
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

from sklearn.model_selection import train_test_split
import pandas as pd
train_path = r'train_df.csv'

train_df = pd.read_csv(train_path)
train_df = train_df.dropna()
train_df = train_df.drop(['Customer_ID'], axis=1)


convert_dict={'Poor':0,'Standard':1,'Good':2}
for (label,num) in convert_dict.items():
    train_df.loc[train_df.index[train_df.loc[:,'Credit_Score']==label],'Credit_Score']=float(num)

# train_df['Credit_Score'] = train_df['Credit_Score'].astype('float32')


X = train_df.drop(['Credit_Score'], axis=1)
y = train_df['Credit_Score']
y = pd.get_dummies(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

# define the keras model

model = Sequential()
model.add(Dense(100, activation='leaky_relu', input_dim=56))
model.add(Dropout(0.5))
model.add(Dense(100, activation='leaky_relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

# model.add(Dense(1, activation='softmax'))
# compile the keras model
# tf.keras.optimizers.Adagrad
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), metrics=['accuracy'])
# fit the keras model on the dataset
history = model.fit(X_train, y_train, epochs=20, batch_size=1000)
# evaluate the keras model
_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))

predict = model.predict(X_val)

a = pd.DataFrame(predict)
b = a.copy()
for i in b:
    if b[0]
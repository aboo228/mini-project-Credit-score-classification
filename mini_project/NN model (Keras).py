import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
# import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,StandardScaler,PowerTransformer
from tqdm import tqdm
# tf.reset_default_graph()

train_df = pd.read_csv(r'C:\Users\sagir\PycharmProjects\craditcard_jupyter_notebook\done_eda_df.csv')


# train_path = r'train_df.csv'
#
# train_df = pd.read_csv(train_path)
# train_df = train_df.dropna()
# train_df = train_df.drop(['Customer_ID'], axis=1)
#
# index_target_s = train_df[train_df['Credit_Score'] == 1].index
# train_df = train_df.drop(index_target_s[:50000], axis=0)

#
# convert_dict={'Poor':0,'Standard':1,'Good':2}
# for (label,num) in convert_dict.items():
#     train_df.loc[train_df.index[train_df.loc[:,'Credit_Score']==label],'Credit_Score']=float(num)
#
# train_df['Credit_Score'] = train_df['Credit_Score'].astype('int32')
#
#
# X = train_df.drop(['Credit_Score'], axis=1)
# y = train_df['Credit_Score']
# y = pd.get_dummies(y)
def prepare_df(df=None, new_data=False):
    drop_features = ['Customer_ID', 'Credit_Score', 'Month']
    if new_data:
        df['Credit_History_Age'] = convert_str_num('Credit_History_Age', [0, 3], [12, 1])
        convert_loans_type(df=df)
        drop_features = ['ID', 'Name', 'SSN', 'Customer_ID', 'Credit_Score', 'Month']
    else:
        pass

    df.Credit_Mix = df.Credit_Mix.map({'Bad': 0, 'Standard': 1, 'Good': 2})
    df.drop(drop_features, axis=1, inplace=True)
    df = pd.get_dummies(train_df, drop_first=True)

    return df


target_dict = {}
for i, x in enumerate(train_df['Credit_Score'].unique()):
    target_dict[x] = i

target = train_df['Credit_Score'].map(target_dict)
train_df = prepare_df(df=train_df)

from imblearn.over_sampling import ADASYN

balanced_df = train_df.copy()
k = round(np.sqrt(balanced_df.shape[1]))
if k % 2 == 0:
    k += 1

balanced_df, target = ADASYN(n_neighbors=k - 2).fit_resample(balanced_df, target)

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(balanced_df, target, test_size=0.25, random_state=42, shuffle=True,
                                                  stratify=target)
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_val)
# define the keras model

# scaler = RobustScaler()
scaler = PowerTransformer(method='yeo-johnson')
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-1,
        decay_steps=2000,
        decay_rate=0.9)
model = Sequential()
model.add(Dense(32*2, activation='sigmoid', input_shape=(48,)))
# model.add(Dropout(0.2))
model.add(Dense(32*5, activation='sigmoid'))
# model.add(Dropout(0.3))
# model.add(Dense(120))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='sigmoid'))
# model.add(Dropout(0.5))
# model.add(Dense(80))
model.add(Dense(3, activation='softmax'))

# model.add(Dense(1, activation='softmax'))
# compile the keras model
# tf.keras.optimizers.Adagrad
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2000)
# fit the keras model on the dataset
history = model.fit(X_train, y_train, epochs=10000, batch_size=X_train.shape[0],callbacks=[callback])

# evaluate the keras model
_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy * 100))

predict = model.predict(X_val)

a = pd.DataFrame(predict)
b = a[0].copy()

for i in tqdm(range(len(a))):
    if (a[1][i] > a[0][i] and a[1][i] > a[2][i]) == True:
        b[i] = 1
    elif (a[0][i] > a[1][i] and a[0][i] > a[2][i]) == True:
        b[i] = 0
    else:
        b[i] = 2
b.groupby(b).count()

model.save(r'C:\Users\sagir\PycharmProjects\craditcard_jupyter_notebook\nnmodel')
# new_df = pd.DataFrame(predict).idxmax(axis=1)
# new_df.groupby(new_df).count()
# test_list = [1,2,3]

# train_df.groupby(train_df['Credit_Score']).count()
# train_df.sort_values(by=['Credit_Score'])
history_df=pd.DataFrame.from_dict(history.history)
sns.scatterplot(history_df)
sns.lineplot(history_df)
plt.show()

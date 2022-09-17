import numpy as np
import tensorflow as tf
from tensorflow import keras


import pandas as pd
from sklearn.model_selection import train_test_split
# layer = keras.layers.Dense(3)
# layer.build((None, 4))  # Create the weights
#
# print("weights:", len(layer.weights))
# print("trainable_weights:", len(layer.trainable_weights))
# print("non_trainable_weights:", len(layer.non_trainable_weights))
#
#
# layer = keras.layers.BatchNormalization()
# layer.build((None, 4))  # Create the weights
#
# print("weights:", len(layer.weights))
# print("trainable_weights:", len(layer.trainable_weights))
# print("non_trainable_weights:", len(layer.non_trainable_weights))
#
#
# layer = keras.layers.Dense(3)
# layer.build((None, 4))  # Create the weights
# layer.trainable = False  # Freeze the layer
#
# print("weights:", len(layer.weights))
# print("trainable_weights:", len(layer.trainable_weights))
# print("non_trainable_weights:", len(layer.non_trainable_weights))
#
# # Make a model with 2 layers
# layer1 = keras.layers.Dense(3, activation="relu")
# layer2 = keras.layers.Dense(3, activation="sigmoid")
# model = keras.Sequential([keras.Input(shape=(3,)), layer1, layer2])
#
# # Freeze the first layer
# layer1.trainable = False
#
# # Keep a copy of the weights of layer1 for later reference
# initial_layer1_weights_values = layer1.get_weights()
#
# # Train the model
# model.compile(optimizer="adam", loss="mse")
# model.fit(np.random.random((2, 3)), np.random.random((2, 3)))
#
# # Check that the weights of layer1 have not changed during training
# final_layer1_weights_values = layer1.get_weights()
# np.testing.assert_allclose(
#     initial_layer1_weights_values[0], final_layer1_weights_values[0]
# )
# np.testing.assert_allclose(
#     initial_layer1_weights_values[1], final_layer1_weights_values[1]
# )
#
#
# inner_model = keras.Sequential(
#     [
#         keras.Input(shape=(3,)),
#         keras.layers.Dense(3, activation="relu"),
#         keras.layers.Dense(3, activation="relu"),
#     ]
# )
#
# model = keras.Sequential(
#     [keras.Input(shape=(3,)), inner_model, keras.layers.Dense(3, activation="sigmoid"),]
# )
#
# model.trainable = False  # Freeze the outer model
#
# assert inner_model.trainable == False  # All layers in `model` are now frozen
# assert inner_model.layers[0].trainable == False  # `trainable` is propagated recursively

train_path = r'train_df.csv'

train_df = pd.read_csv(train_path)
train_df = train_df.dropna()

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


# pca = PCA(n_components=54)
# pca.fit(X)
# X = pca.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)


base_model = keras.applications.Xception(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top.

base_model.trainable = False

inputs = keras.Input(shape=(150, 150, 3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = base_model(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])
model.fit(X_train, y_train, epochs=20)

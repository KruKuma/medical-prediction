import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint

df = pd.read_csv('https://cainvas-static.s3.amazonaws.com/media/user_data/cainvas-admin/diabetes.csv')
df.head()

df.info()

df.duplicated().sum()
df.drop_duplicates(inplace=True)

df.info()

columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in columns:
    df[col].replace(0, np.NaN, inplace=True)

df.info()

df.dropna(inplace=True)
df.info()

X = df.drop('Outcome', axis=1)
X = StandardScaler().fit_transform(X)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

model = Sequential()
model.add(Dense(8, activation = 'relu', input_shape = X_train[0].shape))

model.add(Dense(8, activation='relu'))

model.add(Dense(4, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer= 'adam' ,loss='binary_crossentropy',metrics=['acc'])

checkpointer = ModelCheckpoint('diabetes.h5', monitor='val_acc', mode='max', verbose=2, save_best_only=True)
history=model.fit(X_train, y_train, batch_size=16, epochs=350, validation_data=(X_test, y_test), callbacks=[checkpointer])

present_model = keras.models.load_model('diabetes.h5')
present_model.summary()

print("Accuracy of our model on test data : " , present_model.evaluate(X_test,y_test)[1]*100 , "%")

model.summary()
score = model.evaluate(X_test, y_test, verbose=0)
print('Model Accuracy = ',score[1])

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

open("model.tflite", "wb").write(tflite_model)

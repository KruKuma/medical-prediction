# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as ss
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

dataset = pd.read_csv('extras/diabetes.csv', header=None)

dataset.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                   'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']


# Data Cleaning
dataset.duplicated().sum()
dataset.drop_duplicates(inplace=True)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_validation, y_train, Y_validation = train_test_split(X, y, test_size=0.2, random_state=0)

sc = ss()
X_train = sc.fit_transform(X_train)
X_validation = sc.transform(X_validation)

# SVM
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_validation)

cm_test = confusion_matrix(y_pred, Y_validation)

y_pred_train = model.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print('Accuracy for training set for svm = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for svm = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(Y_validation)))

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(30, input_dim=13, activation='tanh'))
model.add(Dense(20, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=1)

model.summary()
score = model.evaluate(X_validation, Y_validation, verbose=0)
print('Model Accuracy = ',score[1])

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

open("extras/model.tflite", "wb").write(tflite_model)

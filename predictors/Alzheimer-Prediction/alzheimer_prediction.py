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

dataset = pd.read_csv('extras/oasis_longitudinal.csv')
dataset.head()

# dataset Cleaning
dataset = dataset.loc[dataset['Visit'] == 1]  # Only look at first visit
dataset = dataset.reset_index(drop=True)  # reset index after filtering

dataset = dataset[['Group', 'M/F', 'Age', 'EDUC', 'SES',
             'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']]
dataset.rename(columns={'M/F': 'Gender'}, inplace=True)
dataset.head()

# Check for missing values
dataset.isna().sum()
dataset['SES'].value_counts()
dataset['SES'] = dataset['SES'].fillna(2.0)
dataset.isna().sum().sum()

# Binary encode object columns
dataset['Group'] = dataset['Group'].apply(lambda x: 1 if x == 'Demented' else 0)
dataset['Gender'] = dataset['Gender'].apply(lambda x: 1 if x == 'M' else 0)

dataset.head(10)
dataset = dataset.astype('float64')

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
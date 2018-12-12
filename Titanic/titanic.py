#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 01:17:35 2018

@author: Ashish_Trivedi
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#dropping columns which are not necessary
train_data = train_data.drop(['PassengerId','Name', 'Ticket', 'Cabin'], axis = 1)
PassengerId = test_data['PassengerId']
test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)

#change index to passengerid
#rain_data = train_data.set_index('PassengerId')


#to convert features to numeric values
le = preprocessing.LabelEncoder()
columns = ['Sex', 'Embarked']

for col in columns:
    train_data[col] = le.fit_transform(train_data[col].astype(str))
    test_data[col] = le.fit_transform(test_data[col].astype(str))


#filling NaN values with column mean values
train_data.fillna(train_data.mean(), inplace = True)
test_data.fillna(test_data.mean(), inplace = True)


#Preparing train and test data
array = train_data.values
X = array[ : , 1 :]
Y = array[ : , 0]

train_x, test_x, train_y, test_y = model_selection.train_test_split(X, Y, test_size = 0.20, random_state = 6)


#accuracy scores for various models
models = []
#models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis(solver = 'lsqr')))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
#models.append(('RF', RandomForestRegressor(random_state = 1)))
#models.append(('XGB', XGBClassifier()))

#cross-validation scores
'''
results = []
for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state = 7)
    cv_results = model_selection.cross_val_score(model, train_x, train_y, cv = kfold, scoring = 'accuracy')
    results.append(cv_results)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

LR: 0.795051 (0.054102)
LDA: 0.793623 (0.048340)
KNN: 0.699472 (0.058252)
CART: 0.792195 (0.037793)
NB: 0.790865 (0.063504)
SVM: 0.709390 (0.038268)
'''
#Accuracy scores
'''
for name, model in models:
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    print(name, accuracy_score(test_y, pred))
    

LR 0.7988826815642458   
LDA 0.7932960893854749
KNN 0.6983240223463687
CART 0.7597765363128491
NB 0.7597765363128491
SVM 0.6927374301675978
'''

#generating output files for different models to compare scores
for name, model in models:
    model.fit(train_x, train_y)
    pred = model.predict(test_data).astype(int)
    #print(accuracy_score(test_y, pred))
    #submit = pd.DataFrame({'PassengerId': PassengerId, 'Survived' : pred})
    #submit.to_csv(name + 'titanic_submission.csv', index = False)


'''
Scores -
LR - 0.75598
CART - 0.72248
KNN - 0.64593
LDA - 0.76076
'''

submit = pd.DataFrame({'PassengerId': PassengerId, 'Survived' : pred})
submit.to_csv('titanic_submission.csv', index = False)









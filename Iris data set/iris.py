#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#imports for different models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names = colnames)

#splitting data for testing and training
array = dataset.values
x = array[:, 0:4]
y = array[:, 4]
validation_size = 0.20
seed = 7
scoring = 'accuracy'
x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size = validation_size, random_state = seed)

#test models and get accuracy
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s : %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

'''  
LR : 0.966667 (0.040825)
LDA : 0.975000 (0.038188)
KNN : 0.983333 (0.033333)
CART : 0.975000 (0.038188)
NB : 0.975000 (0.053359)
SVM : 0.991667 (0.025000)
'''


# Compare Algorithms
'''
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
'''

#Training and validating SVC model
'''
clf = SVC(gamma = 'auto')
clf.fit(x_train, y_train)
predictions = clf.predict(x_validation)
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))
'''

#Training and validation for all models
for name, model in models:
    clf = model
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_validation)
    print(name, accuracy_score(y_validation, predictions))






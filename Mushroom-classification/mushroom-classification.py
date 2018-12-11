#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 17:50:23 2018

@author: Ashish_Trivedi
"""

import numpy as np
import pandas as pd
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

mdata = pd.read_csv('mushrooms.csv')

#checks for null value
mdata.isnull().sum()


#Auto-labels for each bar color. 
def autolabel(rects,fontsize=14):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1*height,'%d' % int(height),
                ha='center', va='bottom',fontsize=fontsize)


#plot of cap-color
ind = np.arange(10)
ptotal = []
etotal = []
cap_color_total = mdata['cap-color'].value_counts()
cap_color_labels = cap_color_total.axes[0].tolist()
for cap_color in  cap_color_labels:
    e = mdata[(mdata['cap-color'] == cap_color) & (mdata['class'] == 'e')].shape[0]
    etotal.append(e)
    p = mdata[(mdata['cap-color'] == cap_color) & (mdata['class'] == 'p')].shape[0]
    ptotal.append(p)

fig, ax = plt.subplots(figsize = (10,7))
bare1 = ax.bar(ind, etotal, width = 0.4, align = 'center', color = 'r')
barp1 = ax.bar(ind + 0.4, ptotal, width = 0.4, align = 'center', color = 'b')
autolabel(barp1, 6)
autolabel(bare1, 6)
plt.title('cap-color')
plt.xticks(ind + 0.2)
ax.set_xticklabels(('brown' , 'buff' , 'cinnamon', 'gray', 'green', 'pink', 'purple', 'red', 'white', 'yellow'), fontsize = 12)
ax.legend((bare1,barp1),('edible','poisonous'),fontsize=17)
plt.show()

#plot of cap-shape
ind = np.arange(6)
ptotal = []
etotal = []
cap_shape_total = mdata['cap-shape'].value_counts()
cap_shape_labels = cap_shape_total.axes[0].tolist()
for cap_shape in  cap_shape_labels:
    e = mdata[(mdata['cap-shape'] == cap_shape) & (mdata['class'] == 'e')].shape[0]
    etotal.append(e)
    p = mdata[(mdata['cap-shape'] == cap_shape) & (mdata['class'] == 'p')].shape[0]
    ptotal.append(p)

fig, ax = plt.subplots(figsize = (10,7))
bare2 = ax.bar(ind, etotal, width = 0.4, align = 'center', color = 'r')
barp2 = ax.bar(ind + 0.4, ptotal, width = 0.4, align = 'center', color = 'b')
autolabel(barp2, 8)
autolabel(bare2, 8)
plt.xticks(ind + 0.2)
ax.set_xticklabels(('bell', 'conical', 'convex', 'flat', 'knobbed', 'sunken'), fontsize = 12)
ax.legend((bare2,barp2),('edible','poisonous'),fontsize=17)
plt.title('cap-shape')
plt.show()


#to convert features to numeric values
le = preprocessing.LabelEncoder()
for col in mdata.columns:
    mdata[col] = le.fit_transform(mdata[col])
    

array = mdata.values
X = array[:, 1:]
Y = array[: , 0]


#splits train and test data in 80% 20%
train_x, test_x, train_y, test_y = model_selection.train_test_split(X, Y, train_size = 0.20, random_state = 1)

#accuracy scores for various models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RF', RandomForestRegressor(random_state = 1)))
results = []
names = []

for name, model in models:
    clf = model
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    names.append(name)
    results.append(accuracy_score(test_y, pred.round()))
    print(name, accuracy_score(test_y, pred.round()))



'''
LR 0.9535384615384616
LDA 0.9536923076923077
KNN 0.9918461538461538
CART 0.9987692307692307
NB 0.9
SVM 0.9972307692307693
RF 0.9987692307692307 
'''

#model accuracy plots
fig, ax = plt.subplots()
bar3 = ax.bar(names, results, width = 0.6)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.show()



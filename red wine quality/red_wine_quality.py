
# coding: utf-8

# In[1]:


# Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Supress warnings
import warnings
warnings.filterwarnings("ignore")

# Classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier

# Regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor 
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# Modelling Helpers :
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV , KFold , cross_val_score, ShuffleSplit, cross_validate

# Preprocessing :
from sklearn.preprocessing import MinMaxScaler , StandardScaler, Imputer, LabelEncoder

# Metrics :
# Regression
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 
# Classification
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, classification_report

print("Setup complete...")


# In[2]:


rw = pd.read_csv('../input/winequality-red.csv')
print("Dataset loaded...")


# In[3]:


rw.describe()


# In[4]:


rw.columns


# In[5]:


rw.dtypes


# In[6]:


corr = rw.corr()
plt.figure(figsize = (12,12))
sns.heatmap(data = corr, annot = True, square = True, cbar = True)


# # Quality of red wine has
# #### positive corr with -
# * Alcohol content
# * Suplahtes
# * Citric acid
# 
# #### negative correlation with
# * Volatile acidity
# * Total sulphur dioxide

# In[7]:


sns.lineplot(x = 'quality', y = 'alcohol', data = rw)


# In[8]:


sns.lineplot(x = 'quality', y = 'sulphates', data = rw)


# In[9]:


sns.lineplot(x = 'quality', y = 'citric acid', data = rw)


# In[10]:


sns.lineplot(x = 'quality', y = 'volatile acidity', data = rw)


# In[11]:


sns.lineplot(x = 'quality', y = 'total sulfur dioxide', data = rw)


# In[12]:


sns.lineplot(x = 'quality', y = 'pH', data = rw)


# In[13]:


# classifying wines as good or bad
# < 6.5 bad
bins = (2, 6.5, 8)
name = ['bad', 'good']
rw['quality'] = pd.cut(rw['quality'], bins = bins, labels = name)
rw.head(5)


# In[14]:


rw['quality'].value_counts()


# In[15]:


sns.countplot(rw['quality'])


# In[16]:


le = LabelEncoder()
rw['quality'] = le.fit_transform(rw['quality'])


# In[17]:


# Generating data for test and train cases
X = rw.drop(['quality'], axis = 1)
Y = rw['quality']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 7)


# In[18]:


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# # DecisionTreeClassifier

# In[19]:


dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
pred = dtc.predict(x_test)
print(accuracy_score(y_test, pred))


# # KNeighborsClassifier

# In[20]:


knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
print(accuracy_score(pred, y_test))


# # RandomForestClassifier

# In[21]:


rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(x_train, y_train)
pred = rfc.predict(x_test)
print(accuracy_score(y_test, pred))


# # XGBoost

# In[22]:


xgb = XGBClassifier()
xgb.fit(x_train, y_train)
pred = xgb.predict(x_test)
print(accuracy_score(y_test, pred))


# # SVC

# In[23]:


svc = SVC()
cv = ShuffleSplit(n_splits = 2, test_size = 0.20, random_state = 7)
scores = cross_val_score(svc, x_train, y_train, cv = cv)
print(scores.mean())


# # GradientBoostingRegressor

# In[24]:


gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train)
pred = gbr.predict(x_test).astype(int)
print(accuracy_score(y_test, pred))


# # AdaBoostRegressor

# In[25]:


abr = AdaBoostRegressor()
abr.fit(x_train, y_train)
pred = abr.predict(x_test).astype(int)
print(accuracy_score(y_test, pred))


# # Tuning hyperparametres for SVC

# In[26]:


# finding best paramentres for SVC model through GridSearchCV
#Finding best parameters for our SVC model

svc = SVC()

params = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}

clf = GridSearchCV(svc, param_grid = params, scoring = 'accuracy', cv = 10)

clf.fit(x_train, y_train)
clf.best_params_


# In[30]:


# Re-running model with best parametres
svc1 = SVC(C = 1.2, gamma = 1.4, kernel = 'rbf')
svc1.fit(x_train, y_train)
pred = svc1.predict(x_test)
print(accuracy_score(y_test, pred))


# # Tuning hyperparametres for RandomForestClassifier

# In[31]:


rfc = RandomForestClassifier()

params = { 
    'n_estimators': [100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

rfc_clf = GridSearchCV(rfc, param_grid = params, scoring = 'accuracy', cv = 10)
rfc_clf.fit(x_train, y_train)
rfc_clf.best_params_


# In[32]:


rfc1 = RandomForestClassifier(criterion = 'entropy', max_depth = 8, max_features = 'log2', n_estimators = 200)
rfc1.fit(x_train, y_train)
pred = rfc1.predict(x_test)
print(accuracy_score(y_test, pred))


# # Tuning hyperparametres for XGBoost

# In[40]:


# Takes really long to run
'''
xgb = XGBClassifier()
params = {
        'learning_rate' : [0.05, 0.1, 0.5, 1, 2, 3],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

xgb_clf = GridSearchCV(xgb, param_grid = params, scoring = 'accuracy', cv = 10, verbose = 2)
xgb_clf.fit(x_train, y_train)
print(xgb_clf.best_params_)
'''


# In[ ]:


# Best params - {'colsample_bytree': 0.6, 'gamma': 1.5, 'learning_rate': 0.5, 'max_depth': 5, 'min_child_weight': 1, 'subsample': 0.8}
xgb1 = XGBClassifier(learning_rate = 0.5, max_depth = 5, min_child_weight = 1, subsample = 0.8, gamma = 1.5, colsample_bytree = 0.6)
xgb1.fit(x_train, y_train)
pred = xgb1.predict(x_test)
print(accuracy_score(y_test, pred))


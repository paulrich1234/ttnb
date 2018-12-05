# -*- coding: utf-8 -*-
from __future__ import print_function
from sklearn import preprocessing
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


#读取数据
# data =pd.read_excel('origin_data.xlsx')
# X=data.drop(['糖尿病'],axis=1)

# y=data['糖尿病']
data=pd.read_table('../combine_local_net.txt',sep='\t',encoding='utf-8',engine='python')
X=data.drop(['Unnamed: 0','label'],axis=1)
y=data['label']
# print(X)
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# print(y_test)








clf1=XGBClassifier(booster="gbtree", colsample_bytree= 0.9, gamma= 0.2, learning_rate= 0.3, max_depth= 4, min_child_weight= 4,
                  n_estimators= 500, objective= "binary:logistic", seed= 0)

clf1.fit(X_train,y_train)

y_true, y_pred = y_test, clf1.predict(X_test)

print(classification_report(y_true, y_pred))
print()


clf2 =RandomForestClassifier(n_estimators=200)
clf2.fit(X_train,y_train)

y_true, y_pred = y_test, clf2.predict(X_test)

print(classification_report(y_true, y_pred))
print()


clf3 =SVC(gamma='auto')
clf3.fit(X_train,y_train)

y_true, y_pred = y_test, clf3.predict(X_test)

print(classification_report(y_true, y_pred))
print()

clf4=LogisticRegression()
clf4.fit(X_train,y_train)
y_true, y_pred = y_test, clf4.predict(X_test)

print(classification_report(y_true, y_pred))
print()


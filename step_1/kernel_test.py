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
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt




# # 训练数据标准化
# min_max_scaler = preprocessing.MinMaxScaler()
# data=pd
# X_train=train.drop(['label'],axis=1)
# Y_train=train['label']
# X_train_minmax = min_max_scaler.fit_transform(X_train)
# # train_data =pd.concat([a,Y_train],axis=1,join_axes=[a.index])
# # print(train_data)

data=pd.read_table('../combine_local_net.txt',sep='\t',encoding='utf-8',engine='python')
X=data.drop(['Unnamed: 0','label'],axis=1)
X=np.array(X)

y=data['label']



scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=35)
# 其中gamma的选值很重要，默认为1/n_features
X_skernpca = scikit_kpca.fit_transform(X)
print(X_skernpca)

print(X_skernpca[y==0, 0])

print('*'*100)

print(X_skernpca[y==0, 1])


fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))

ax[0].scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1],color='red', marker='^', alpha=0.5)
ax[1].scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1],color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

plt.tight_layout()
plt.show()


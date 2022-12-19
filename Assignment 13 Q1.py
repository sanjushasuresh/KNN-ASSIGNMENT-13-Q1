# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 12:10:11 2022

@author: LENOVO
"""

import numpy as np
import pandas as pd

df=pd.read_csv("glass.csv")
df.head()
df.shape
df.duplicated()
df[df.duplicated()]
df.drop([39], axis=0, inplace=True)
df.info()
df.corr()
# Type and Mg has high negative correlation and there is a high correlation b/w Ri and Ca i.e. 0.811183


# Splitting the variables
X=df.iloc[:,0:9]
X1=df.iloc[:,1:9]
Y=df["Type"]

# Standardization
from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()

# X1 columns Standidization

X1["Na"]=MM.fit_transform(X1[["Na"]])

X1["Mg"]=MM.fit_transform(X1[["Mg"]])

X1["Al"]=MM.fit_transform(X1[["Al"]])

X1["Si"]=MM.fit_transform(X1[["Si"]])

X1["K"]=MM.fit_transform(X1[["K"]])

X1["Ca"]=MM.fit_transform(X1[["Ca"]])

X1["Ba"]=MM.fit_transform(X1[["Ba"]])

X1["Fe"]=MM.fit_transform(X1[["Fe"]])


# Model 1 fitting using X1

from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=5)
KNN.fit(X1,Y)
Y_predict=KNN.predict(X1)

from sklearn.metrics import accuracy_score
as1=accuracy_score(Y,Y_predict)
print(as1) # 0.76056
# Accuracy is 76%

from sklearn.model_selection import KFold, cross_val_score
k=10
k_fold=KFold(n_splits=k, random_state=None)
cv_scores=cross_val_score(KNN, X1, Y, cv=k_fold)
mean_acc_score=sum(cv_scores)/len(cv_scores)
# Mean accuracy score = 78%

from sklearn.metrics import roc_curve, roc_auc_score
KNN.predict_proba(X1)[:,1]
fpr, tpr, threshold  = roc_curve(Y,KNN.predict_proba(X1)[:,1], pos_label=1)
import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.ylabel('tpr - True Positive Rate')
plt.xlabel('fpr - False Positive Rate')
plt.show()

aucvalue = roc_auc_score(Y,KNN.predict_proba(X1), multi_class='ovo')
print("aucvalue", aucvalue.round(3))
# Aucvalue = 0.963

#=============================================================================================#


from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()

# X columns Standidization

X["RI"]=MM.fit_transform(X[["RI"]])

X["Na"]=MM.fit_transform(X[["Na"]])

X["Mg"]=MM.fit_transform(X[["Mg"]])

X["Al"]=MM.fit_transform(X[["Al"]])

X["Si"]=MM.fit_transform(X[["Si"]])

X["K"]=MM.fit_transform(X[["K"]])

X["Ca"]=MM.fit_transform(X[["Ca"]])

X["Ba"]=MM.fit_transform(X[["Ba"]])

X["Fe"]=MM.fit_transform(X[["Fe"]])


# Model 2 fitting using X

from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=5)
KNN.fit(X,Y)
Y_predict=KNN.predict(X)

from sklearn.metrics import accuracy_score
as2=accuracy_score(Y,Y_predict)
print(as2) # 0.78873
# Accuracy is 79%

from sklearn.model_selection import KFold, cross_val_score
k=10
k_fold=KFold(n_splits=k, random_state=None)
cv_scores=cross_val_score(KNN, X, Y, cv=k_fold)
mean_acc_score=sum(cv_scores)/len(cv_scores)
# Mean accuracy score = 80%

from sklearn.metrics import roc_curve, roc_auc_score
KNN.predict_proba(X)[:,1]
fpr, tpr, threshold  = roc_curve(Y,KNN.predict_proba(X)[:,1], pos_label=1)
import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.ylabel('tpr - True Positive Rate')
plt.xlabel('fpr - False Positive Rate')
plt.show()

aucvalue = roc_auc_score(Y,KNN.predict_proba(X), multi_class='ovo')
print("aucvalue", aucvalue.round(3))
# Aucvalue = 0.969
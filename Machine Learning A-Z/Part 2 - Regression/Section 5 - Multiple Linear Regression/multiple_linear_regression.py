#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:08:21 2019

@author: diptu
"""
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import datasetx
df = pd.read_csv('50_Startups.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:,4].values

#Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Split data into train and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Fitting the Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

#Predicting test result
y_pred = lr.predict(X_test)

#Building optimal model using Backward Elemination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

X_opt = X[:, [0,1,2,3,4,5,6]]
OLS = sm.OLS(endog=y, exog=X_opt ).fit()
OLS.summary()


X_opt = X[:, [0,1,2,3,4,6]]
OLS = sm.OLS(endog=y, exog=X_opt ).fit()
OLS.summary()


X_opt = X[:, [0,1,2,3,4]]
OLS = sm.OLS(endog=y, exog=X_opt ).fit()
OLS.summary()
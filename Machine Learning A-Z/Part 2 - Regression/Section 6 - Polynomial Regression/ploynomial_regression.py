#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 20:57:24 2019

@author: diptu
"""
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import datasetx
df = pd.read_csv('Position_Salaries.csv')

X = df.iloc[:,1:2].values #X-is now a matrix
y = df.iloc[:,2].values # y is a vector

"""
This dataset is too small to split into training and test set
"""
#Linear Regration model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)

#Polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=4)
X_poly = poly_features.fit_transform(X)

poly_lr = LinearRegression()
poly_lr.fit(X_poly,y)

#Visualization of Linear Regression
plt.scatter(x=X,y=y,color = 'red')
plt.plot(X,lr.predict(X), color='blue')
plt.title('Liner Regression')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

#Visualization of Polynomial Regression
plt.scatter(x=X,y=y,color = 'red')
plt.plot(X,poly_lr.predict(poly_features.fit_transform(X)), color='blue')
plt.title('Ploynomial Regression')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

lr.predict(6.5)
poly_lr.predict(poly_features.fit_transform(6.5))
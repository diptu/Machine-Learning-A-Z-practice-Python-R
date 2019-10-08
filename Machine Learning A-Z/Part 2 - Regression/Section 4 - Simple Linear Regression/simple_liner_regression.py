#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 00:20:38 2019

@author: diptu
"""
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import datasetx
df = pd.read_csv('Salary_Data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:,1].values

#Split data into train and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Fitting Simple Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

#Predict test result
y_pred = lr.predict(X_test)

#Visualization the training set result
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, lr.predict(X_train), color='blue')
plt.title('Salary vs Experience (Train set)')
plt.xlabel('Salary in USD')
plt.ylabel('Experience(year)')
plt.show()


#Visualization the test set result
plt.scatter(X_test, y_test, color='green')
plt.plot(X_train, lr.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Salary in USD')
plt.ylabel('Experience(year)')
plt.show()







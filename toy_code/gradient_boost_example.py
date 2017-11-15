#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn import tree
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

# #############################################################################
# Load data
boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# calculate the Mse loss
def loss(y,y_pred):
    return mean_squared_error(y,y_pred)

# find argmin gamma in respect of loss
def argmin_loss(y,Fmm1,h):
    g = 2 * (Fmm1-y)/h
    return g

sam_num = X_train.shape[0]
maxiter = 10
global gamma
resi = np.zeros([maxiter,sam_num])

def Fun_mm1(X):
    sam_num = X.shape[0]
    return np.zeros(sam_num)
def Fun_m(X):
    sam_num = X.shape[0]
    return np.mean(y_train) * np.ones(sam_num)

for i in range(maxiter):
    Fm = Fun_m(X_train)
    Fmm1 = Fun_mm1(X_train)
    dF = Fm - Fmm1
    dloss = loss(y_train,Fm) - loss(y_train,Fmm1)
    res = - np.divide(dloss,dF)
    tree_params = {'max_depth': 2, 'min_samples_split': 2}
    iter_tree = tree.DecisionTreeRegressor(**tree_params);
    iter_tree.fit(X_train,y_train)
    hx = iter_tree.predict(X_train)
    gamma = argmin_loss(y_train,Fm,hx)
    
    
    
    
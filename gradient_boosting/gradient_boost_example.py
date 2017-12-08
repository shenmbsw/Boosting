#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Nov 15 17:26:24 2017
@author: Shen Shen
This is a demo code for EC503 Project
"""

import numpy as np
from sklearn import datasets
from sklearn import tree
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


class Gradiant_boost():
    def __init__(self, structure_params,tree_params):
        self.loss_array = []
        self.gamma_array = []
        self.tree_array = []
        self.maxiter = structure_params['max_iter']
        self.tree_params = tree_params
        self.learning_rate = structure_params['learning_rate']

    # calculate the Mse loss
    def loss(self,y,y_pred):
        return (y - y_pred) ** 2

    # find argmin gamma in respect of loss
    # using exhaused sreach here
    # this is a convex optimization problem
    def argmin_loss(self,y,Fmm1,h):
        all_loss = []
        all_g = np.linspace(-10.0, 10.0, num=500)
        for g in all_g:
            y_pred = Fmm1 + g*h
            this_loss = np.sum(self.loss(y,y_pred))
            all_loss.append(this_loss)
        idx = np.argmin(all_loss)
        return all_g[idx]

    # find pseudo_residuals
    def pseudo_residuals(self,y,Fmm1,Fm):
        dloss = self.loss(y_train,Fm) - self.loss(y_train,Fmm1)
        df = Fm - Fmm1
        return -dloss/df
    
    # The first tree is to predict the mean of y. which is \gamma * ones(x.shape)
    # if \gamma = y.mean
    def Init_tree(self,X):
        sam_num = X.shape[0]
        return np.ones(sam_num)

    def fit(self,X_train,y_train):
        sam_num = X_train.shape[0]
        Fmm1 = np.zeros(sam_num)
        iter_gamma = np.mean(y_train)
        iter_tree = self.Init_tree
        Fm = self.learning_rate * iter_gamma * iter_tree(X_train)
        iter_loss =  np.sum(self.loss(y_train,Fm))
        self.gamma_array.append(iter_gamma)
        self.tree_array.append(iter_tree)
        self.loss_array.append(iter_loss)

        for i in range(1,self.maxiter):
            res = self.pseudo_residuals(y_train, Fmm1, Fm)
            DTR = tree.DecisionTreeRegressor(**self.tree_params);
            DTR.fit(X_train,res)
            iter_tree = DTR.predict
            hx = iter_tree(X_train)
            iter_gamma = self.argmin_loss(y_train, Fm, hx)
            Fmm1 = Fm
            Fm = Fm + self.learning_rate * iter_gamma * iter_tree(X_train)
            iter_loss = np.sum(self.loss(y_train,Fm))          
            self.gamma_array.append(iter_gamma)
            self.tree_array.append(iter_tree)
            self.loss_array.append(iter_loss)

    def predict(self,X_test):
        max_it = len(self.gamma_array)
        for i in range(max_it):
            if i == 0:
                y_pred = self.learning_rate * self.gamma_array[i] * self.tree_array[i](X_test)
            else:
                y_pred = y_pred + self.learning_rate * self.gamma_array[i] * self.tree_array[i](X_test)
        return y_pred

# Load data
boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# bulid Gradiant boosting model
mi = 300
structure_params = {'max_iter': mi, 'learning_rate': 0.03}
tree_params = {'max_depth': 4, 'min_samples_split': 2}

DGR = Gradiant_boost(structure_params,tree_params)
DGR.fit(X_train,y_train)

train_loss = []
test_loss = []
y_pred = np.zeros(X_test.shape[0])
for i in range(mi):
    y_pred = y_pred + DGR.learning_rate * DGR.gamma_array[i] * DGR.tree_array[i](X_test)
    mse = mean_squared_error(y_test,y_pred)
    test_loss.append(mse)
    train_loss.append(DGR.loss_array[i]/455)

x_axis = np.linspace(1,mi,mi) 
plt.title('loss')
plt.plot(x_axis,train_loss, 'b-',
         label='Training Set Loss')
plt.plot(x_axis,test_loss, 'r-',
         label='Test Set Loss')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('MSE Loss')

train_pred = DGR.predict(X_train)
print('train_loss:%f'%mean_squared_error(y_train,train_pred))
test_pred = DGR.predict(X_test)
print('test_loss:%f'%mean_squared_error(y_test,test_pred))
print('min testing loss:%f'%min(test_loss))



    
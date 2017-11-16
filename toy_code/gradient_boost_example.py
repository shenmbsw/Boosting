#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import datasets
from sklearn import tree
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error


class Gradiant_boost():
    def __init__(self,max_iter):
        self.loss_array = []
        self.gamma_array = []
        self.tree_array = []
        self.maxiter = max_iter
        self.tree_params = {'max_depth': 4, 'min_samples_split': 2}

    # calculate the Mse loss
    def loss(self,y,y_pred):
        return (y - y_pred) ** 2    

    # find argmin gamma in respect of loss
    def argmin_loss(self,y,Fmm1,h):
        # need to be finish
        g = 0.5
        # end here
        return g
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
        Fm = iter_gamma * iter_tree(X_train)
        iter_loss = self.loss(y_train,Fm)
        
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
            Fm = self.predict(X_train)
            iter_loss = self.loss(y_train,Fm)          
            self.gamma_array.append(iter_gamma)
            self.tree_array.append(iter_tree)
            self.loss_array.append(iter_loss)

    def predict(self,X_test):
        max_it = len(self.gamma_array)
        y_pred = np.zeros(X_test.shape[0])
        for i in range(max_it):
            y_pred = y_pred + self.gamma_array[i] + self.tree_array[i](X_test)
        return y_pred

# #############################################################################
# Load data
boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]
    
DGR = Gradiant_boost(10)
DGR.fit(X_train,y_train)
b = DGR.predict(X_test)


    
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
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

class Gradiant_boost():
    def __init__(self, structure_params,tree_params,Adaptive_LR=True):
        self.use_gamma = Adaptive_LR
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
    # using line sreach here
    def argmin_loss(self,y,Fmm1,h):
        all_loss = []
        all_gamma = np.linspace(-10.0, 10.0, num=500)
        for g in all_gamma:
            y_pred = Fmm1 + g*h
            this_loss = np.sum(self.loss(y,y_pred))
            all_loss.append(this_loss)
        idx = np.argmin(all_loss)
        return all_gamma[idx]

    # calculate pseudo_residuals
    def pseudo_residuals(self,y,Fmm1,Fm):
        dloss = self.loss(y,Fm) - self.loss(y,Fmm1)
        df = Fm - Fmm1
        return -dloss/df

    # The first tree is to predict the mean of y. which is gamma * ones(x.shape)
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
            if self.use_gamma:
                Fm = Fm + self.learning_rate * iter_gamma * iter_tree(X_train)
            else:
                Fm = Fm + self.learning_rate * iter_tree(X_train)
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
                if self.use_gamma:
                    y_pred = y_pred + self.learning_rate * self.gamma_array[i] * self.tree_array[i](X_test)
                else:
                    y_pred = y_pred + self.learning_rate  * self.tree_array[i](X_test)
        return y_pred


def get_result(data,mi,tree_depth,adaptive,learning_rate):
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']    
    structure_params = {'max_iter': mi, 'learning_rate': learning_rate}
    tree_params = {'max_depth': tree_depth, 'min_samples_split': 2}
    DGR = Gradiant_boost(structure_params, tree_params, Adaptive_LR=adaptive)
    DGR.fit(X_train,y_train)
    train_loss = []
    test_loss = []
    y_pred = np.zeros(X_test.shape[0])
    train_loss = np.zeros(mi)
    test_loss = np.zeros(mi)
    for i in range(mi):
        if DGR.use_gamma:
            y_pred = y_pred + DGR.learning_rate * DGR.gamma_array[i] * DGR.tree_array[i](X_test)
        else:
            y_pred = y_pred + DGR.learning_rate * DGR.tree_array[i](X_test)
        mse = mean_squared_error(y_test,y_pred)
        test_loss[i] = mse
        train_loss[i] = (DGR.loss_array[i]/455)
    return train_loss, test_loss

def plot_result(mi,tree_depth,adaptive_sl,learning_rate):
    fold_train_loss = np.zeros([10,mi])
    fold_test_loss = np.zeros([10,mi])
    kf = KFold(10)
    i = 0
    for train_idx, test_idx in kf.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        data={'X_train':X_train,'X_test':X_test,
              'y_train':y_train,'y_test':y_test}
        fold_train_loss[i,:],fold_test_loss[i,:] = get_result(data,mi,tree_depth,adaptive_sl,learning_rate)
        i+=1
    train_loss = np.mean(fold_train_loss,0)
    test_loss = np.mean(fold_test_loss,0)
    x_axis = np.linspace(1,mi,mi) 
    plt.title('loss')
    plt.plot(x_axis,train_loss, 'b-',
             label='Training Set Loss')
    plt.plot(x_axis,test_loss, 'r-',
             label='Test Set Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('MSE Loss')
    print('train_loss:%f'%train_loss[-1])
    print('test_loss:%f'%test_loss[-1])
    print('min testing loss:%f'%min(test_loss))

if __name__ == '__main__': 
    # Load data
    boston = datasets.load_boston()
    X, y = shuffle(boston.data, boston.target)
    X = X.astype(np.float32)
    # define the parameter
    mi = 10
    tree_depth = 2
    adaptive_sl = True
    learning_rate = 1
    plot_result(mi,tree_depth,adaptive_sl,learning_rate)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 23:33:24 2019

@author: hys1435
"""

import numpy as np
import pandas as pd
#from KRR_algorithm import compute_mse, compute_mse_no_avg
from sim_study_helper_funs import init_sim_data, init_params
#from process_data import processData
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn import preprocessing
from random import shuffle
import matplotlib.pyplot as plt

#%%

def split_into_m_parts(X, m):
    # split the entire dataset into subsets
    n = int(X.shape[0] / m)
    resShape = [m, n]
    if len(X.shape) > 1:
        for item in X.shape[1:]:
            resShape.append(item)
    res = np.zeros(resShape)
    for i in range(m):
        res[i] = X[i*n:(i+1)*n]
    return res

def gaussianRBF(u, v, sigma):
    return np.exp(-np.einsum('ij,ij->i',u-v,u-v)/(2*sigma**2))

def sobolev(u, v, params):
    return 1 + np.minimum(u, v)

def compute_gram_mat(X1, X2, sigma):
    gram_mat = np.zeros((X1.shape[0],X2.shape[0]))
    for i, item in enumerate(X2):
        gram_mat[:,i] = sobolev(X1, item, sigma)
    return gram_mat

N = 4
m = 1
n = int(N/m)
lam = N**(-2/3)
sigma = -1
X, y = init_sim_data(N)
ind = [x for x in range(N)]
shuffle(ind)
np.take(X, ind, axis = 0, out = X)
np.take(y, ind, axis = 0, out = y)

#X_train_split = split_into_m_parts(X, m) # shape of m, n, d
#y_train_split = split_into_m_parts(y, m)

num = 20
y_pred_lst = np.zeros((N, num))
X_train_1 = X[0:n]
y_train_1 = X[0:n]
K = compute_gram_mat(X_train_1, X_train_1, sigma)
alpha = np.linalg.solve(K + lam * n * np.eye(n), y_train_1)
X_seq = np.linspace(start = 1e-4, stop = 1, num = num)
K_test = compute_gram_mat(X_seq, X_train_1, sigma)
y_pred = K_test @ alpha
for i in range(n):
    y_pred_lst[i] = K_test[:,i]
print(alpha)
#print(y_pred)
#print(y_pred_lst)

# Plot results
fig, ax = plt.subplots()
cols = ['red', 'blue', 'purple', 'orange']
ax.plot(X_seq, y_pred,label='actual function')
for i in range(n):
    ax.plot(X_seq, y_pred_lst[i].astype(float), c=cols[i],
            label='{}th eigenfunction'.format(i+1))
    #ax.set_yscale('log')
plt.legend(loc='upper left')
plt.xlabel("x")
plt.ylabel("Value")
plt.show()

plt.show()
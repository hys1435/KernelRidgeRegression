#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:20:13 2019

@author: zli9
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from random import shuffle
import time
import matplotlib.pyplot as plt
import matplotlib.ticker

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
    
def compute_gram_mat(X1, X2, sigma):
    gram_mat = np.zeros((X1.shape[0],X2.shape[0]))
    for i, item in enumerate(X2):
        gram_mat[:,i] = gaussianRBF(X1, item, sigma)
    return gram_mat

np.random.seed(147)

features = ['year', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't1 9', 't20', 't21', 't22', 't23', 't24', 't25', 't26', 't27', 't28', 't29', 't30', 't31', 't32', 't33', 't34', 't35', 't36 ', 't37', 't38', 't39', 't40', 't41', 't42', 't43', 't44', 't45', 't46', 't47', 't48', 't49', 't50', 't51', 't52', 't53' , 't54', 't55', 't56', 't57', 't58', 't59', 't60', 't61', 't62', 't63', 't64', 't65', 't66', 't67', 't68', 't69', 't70', 't71', 't72', 't73', 't74', 't75', 't76', 't77', 't78', 't79', 't80', 't81', 't82', 't83', 't84', 't85', 't86', 't87', 't88', 't89', 't90']

# Note that our classes (which we have to predict from those 90 features), are all
# the years from 1922 to 2011: 1922, 1923, 1924, 1925, ..., 2011
# Theare exactly 90 years, so we also have 90 classes:
nb_classes = 90

data = pd.read_csv('YearPredictionMSD.csv', names=features)

X = data.ix[:,1:].as_matrix()  # this is the 90 columns without the year
Y = data.ix[:,0].as_matrix()   # this is the year column

X = preprocessing.scale(X)

Y = Y - Y.min()        # The years 1922-2011 are mapped to 0-89

# Training data set
X_train_l = X[0:463715]
y_train_l = Y[0:463715]

# Validation data set
X_test_l = X[463715:]
y_test_l = Y[463715:]

X_train_l = np.array(X_train_l)
X_test_l = np.array(X_test_l)
y_train_l = np.array(y_train_l)
y_test_l = np.array(y_test_l)

N_train_lst = np.array([1024, 2048, 4096, 5120, 6144, 7168, 8192, 10240])
#N_train_lst = np.array([2048, 4096])
for N_train in N_train_lst: 
    N_train = int(N_train)
# N_train = 4096 # small sample test, 2048 produces reasonable results for fast-KRR
    N_test = int(N_train/8)
    X_train = np.array(X_train_l[0:N_train])
    X_test = np.array(X_test_l[0:N_test])
    y_train = np.array(y_train_l[0:N_train])
    y_test = np.array(y_test_l[0:N_test])
    
    mLst = np.array([32, 38, 48, 64, 96, 128, 256])
    sim_num = 10
    mse_lst = np.zeros((mLst.size, sim_num)) # list of mse with under-regularization
    mse_lst_na = np.zeros((mLst.size, sim_num)) # list of mse with under-regularization
    for k in range(sim_num):
        ind_1 = [x for x in range(N_train)]
        shuffle(ind_1)
        ind_2 = [x for x in range(N_test)]
        shuffle(ind_2)
        np.take(X_train, ind_1, axis = 0, out = X_train)
        np.take(X_test, ind_2, axis = 0, out = X_test)
        np.take(y_train, ind_1, axis = 0, out = y_train)
        np.take(y_test, ind_2, axis = 0, out = y_test)
        start_time = time.time()
        for j, m in enumerate(mLst):
            sigma = 32 * np.sqrt(2) # sqrt(2) is for the version the author uses here: 2*sigma**2
            lam = N_train**(-1)
            n = int(N_train/m)
            X_train_split = split_into_m_parts(X_train, m) # shape of m, n, d
            y_train_split = split_into_m_parts(y_train, m)
            
            y_pred_lst = np.zeros((m, N_test))
            for i, (XX, yy) in enumerate(zip(X_train_split, y_train_split)):
                K = compute_gram_mat(XX, XX, sigma)
                alpha = np.linalg.solve(K + lam * n * np.eye(n), yy)
                K_test = compute_gram_mat(X_test, XX, sigma)
                y_pred_lst[i] = K_test @ alpha
            y_pred = np.mean(y_pred_lst, axis = 0)
            mse_lst[j,k] = np.mean((y_test - y_pred)**2)
            X_train_1 = X_train[0:n]
            y_train_1 = y_train[0:n]
            K = compute_gram_mat(X_train_1, X_train_1, sigma)
            alpha = np.linalg.solve(K + lam * m * n * np.eye(n), yy)
            K_test = compute_gram_mat(X_test, X_train_1, sigma)
            y_pred = K_test @ alpha
            mse_lst_na[j,k] = np.mean((y_test - y_pred)**2)
            #print("run time is: ", time.time() - start_time)
    
    
    mse_lst_err = np.std(mse_lst, axis = 1)
    mse_lst = np.mean(mse_lst, axis = 1)
    mse_lst_na_err = np.std(mse_lst_na, axis = 1)
    mse_lst_na = np.mean(mse_lst_na, axis = 1)
    
    print(mse_lst)
    print(mse_lst_err)
    print(mse_lst_na)
    print(mse_lst_na_err)
    
    # Plot results
    fig, ax = plt.subplots()
    cols = ['red', 'blue']
    markers = ['o', 's']
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xticks(mLst)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.errorbar(mLst, mse_lst, yerr = mse_lst_err, c=cols[0], marker=markers[0],label='Fast-KRR',capsize=6)
    ax.errorbar(mLst, mse_lst_na, yerr = mse_lst_na_err, c=cols[1], marker=markers[1],label='KRR with 1/m data',capsize=6) # seeds: 111, 147
    plt.legend(loc='upper left')
    plt.xlabel("log(# of partitions)/log(# of samples)")
    plt.ylabel("Mean square error")
    plt.savefig("N={}".format(N_train))
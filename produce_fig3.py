#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 13:56:14 2019

@author: zli9
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from random import shuffle
import time
import matplotlib.pyplot as plt

#%%
    
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
X_train = X[0:463715]
y_train = Y[0:463715]

# Validation data set
X_test = X[463715:]
y_test = Y[463715:]

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

#%%

# Produce Figure 3
#np.random.seed(147)

def rand_fourier_features(X, D, sigma):
    # d: dimension of data; D: dimension of samples
    d = X.shape[1]
    w = np.random.multivariate_normal(mean = np.zeros(d), cov = 1/sigma**2 * np.eye(d), size = D)
    #print(w.shape)
    b = np.random.uniform(low = 0, high = 2*np.pi, size = D)
    #print(b.shape)
    #print(X.shape)
    z = np.sqrt(2/D) * np.cos(np.matmul(w, np.transpose(X)).transpose() + b)
    return z

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

#N_train = 463715
#N_test = 51630
N_train = 4096 # small sample test
N_test = int(N_train/8)

X_train_s = np.array(X_train[0:N_train])
X_test_s = np.array(X_test[0:int(N_train/8)])
y_train_s = np.array(y_train[0:N_train])
y_test_s = np.array(y_test[0:int(N_train/8)])

m_lst = np.array([16, 24, 32, 48, 64, 96])
D_lst = np.array([100, 600, 1000, 1500, 2000, 2750])
rank_lst = np.array([30, 50, 60, 75, 90, 100])

sim_num = 10
sigma = 45
lam = N_train**(-1)
mse_KRR = np.zeros((len(m_lst), sim_num))
mse_nys = np.zeros((len(rank_lst), sim_num))
mse_ran = np.zeros((len(D_lst), sim_num))
run_time_KRR = np.zeros(len(m_lst))
run_time_Nys = np.zeros(len(rank_lst))
run_time_ran = np.zeros(len(D_lst))

for k in range(sim_num):
    ind_1 = [x for x in range(N_train)]
    shuffle(ind_1)
    ind_2 = [x for x in range(N_test)]
    shuffle(ind_2)
    np.take(X_train_s, ind_1, axis = 0, out = X_train_s)
    np.take(X_test_s, ind_2, axis = 0, out = X_test_s)
    np.take(y_train_s, ind_1, axis = 0, out = y_train_s)
    np.take(y_test_s, ind_2, axis = 0, out = y_test_s)
    

    for j, m in enumerate(m_lst): 
        start_time = time.time()
        n = int(N_train/m)
        X_train_split = split_into_m_parts(X_train_s, m) # shape of m, n, d
        y_train_split = split_into_m_parts(y_train_s, m)
        
        y_pred_lst = np.zeros((m, N_test))
        for i, (XX, yy) in enumerate(zip(X_train_split, y_train_split)):
            K = compute_gram_mat(XX, XX, sigma)
            alpha = np.linalg.solve(K + lam * n * np.eye(n), yy)
            K_test = compute_gram_mat(X_test_s, XX, sigma)
            y_pred_lst[i] = K_test @ alpha
        y_pred = np.mean(y_pred_lst, axis = 0)
        mse_KRR[j,k] = np.mean((y_test_s - y_pred)**2)
        run_time_KRR[j] += time.time() - start_time

    
    for j, D in enumerate(D_lst):
        start_time = time.time()
        d = X_train_s.shape[1]
        w = np.random.multivariate_normal(mean = np.zeros(d), 
                                          cov = 1/sigma**2 * np.eye(d), size = D)
        b = np.random.uniform(low = 0, high = 2*np.pi, size = D)
        Z = np.sqrt(2/D) * np.cos(np.matmul(w, np.transpose(X_train_s)).transpose() + b)
        new_col = np.ones(N_train).reshape(N_train, 1)
        ZZ = np.concatenate((Z, new_col), 1)
        Zt = np.sqrt(2/D) * np.cos(np.matmul(w, np.transpose(X_test_s)).transpose() + b)
        new_col = np.ones(N_test).reshape(N_test, 1)
        ZZt = np.concatenate((Zt, new_col), 1)
        w = np.linalg.solve(ZZ.transpose() @ ZZ + lam * N_train * np.eye(D+1), ZZ.transpose() @ y_train_s) 
        y_pred = ZZt @ w
        mse_ran[j,k] = np.mean((y_test_s - y_pred)**2)
        run_time_ran[j] += time.time() - start_time
    

    for j, rank in enumerate(rank_lst):
        # Nystrom sampling
        start_time = time.time()
        Knm = compute_gram_mat(X_train_s, X_train_s[0:rank], sigma)
        Kmm = compute_gram_mat(X_train_s[0:rank], X_train_s[0:rank], sigma)
        U, D, VT = np.linalg.svd(Kmm, full_matrices=False)
        C = Knm @ U @ np.diag(np.sqrt(D)**(-1))
        mu = lam * N_train
        CT = C.transpose()
        T = CT @ C + mu * np.eye(rank)
        alpha = 1 / mu * (y_train_s - C @ np.linalg.inv(T) @ CT @ y_train_s)
        K_test = compute_gram_mat(X_test_s, X_train_s, sigma)
        y_pred = K_test @ alpha
        mse_nys[j,k] = np.mean((y_test_s - y_pred)**2)
        run_time_Nys[j] += time.time() - start_time

    print("{}th iteration done".format(k))

mse_KRR_err = np.std(mse_KRR, axis = 1)
mse_KRR = np.mean(mse_KRR, axis = 1)
mse_nys_err = np.std(mse_nys, axis = 1)
mse_nys = np.mean(mse_nys, axis = 1)
mse_ran_err = np.std(mse_ran, axis = 1)
mse_ran = np.mean(mse_ran, axis = 1)
#np.savetxt("mse_ran_err.out", mse_ran_err)
#np.savetxt("mse_ran.out", mse_ran)
#np.savetxt("run_time_ran.out", run_time_ran)
#np.savetxt("mse_nys_err.out", mse_nys_err)
#np.savetxt("mse_nys.out", mse_nys)
#np.savetxt("run_time_nys.out", run_time_Nys)
#np.savetxt("mse_KRR_err.out", mse_KRR_err)
#np.savetxt("mse_KRR.out", mse_KRR)
#np.savetxt("run_time_KRR.out", run_time_KRR)

# Plot results
fig, ax = plt.subplots()
cols = ['red', 'blue', 'purple']
markers = ['o', 's', '^']
ax.errorbar(run_time_KRR, mse_KRR, yerr = mse_KRR_err, c=cols[0], marker=markers[0],label='Fast-KRR',capsize=5)
ax.errorbar(run_time_ran, mse_ran, yerr = mse_ran_err, c=cols[1], marker=markers[1],label='Random Feature Approx.',capsize=5) # seeds: 111, 147
ax.errorbar(run_time_Nys, mse_nys, yerr = mse_nys_err, c=cols[2], marker=markers[2],label='Nystrom Sampling',capsize=5)
plt.legend(loc='upper right')
plt.xlabel("time")
plt.ylabel("Mean square error")
plt.savefig("N={} Fig3_avg".format(N_train))
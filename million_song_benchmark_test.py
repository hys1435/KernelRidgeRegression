#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 13:56:14 2019

@author: zli9
"""

import numpy as np
import pandas as pd
#from KRR_algorithm import compute_mse, compute_mse_no_avg
#from sim_study_helper_funs import init_params
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
from random import shuffle
import time
import matplotlib.pyplot as plt
    
features = ['year', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't1 9', 't20', 't21', 't22', 't23', 't24', 't25', 't26', 't27', 't28', 't29', 't30', 't31', 't32', 't33', 't34', 't35', 't36 ', 't37', 't38', 't39', 't40', 't41', 't42', 't43', 't44', 't45', 't46', 't47', 't48', 't49', 't50', 't51', 't52', 't53' , 't54', 't55', 't56', 't57', 't58', 't59', 't60', 't61', 't62', 't63', 't64', 't65', 't66', 't67', 't68', 't69', 't70', 't71', 't72', 't73', 't74', 't75', 't76', 't77', 't78', 't79', 't80', 't81', 't82', 't83', 't84', 't85', 't86', 't87', 't88', 't89', 't90']

# Note that our classes (which we have to predict from those 90 features), are all
# the years from 1922 to 2011: 1922, 1923, 1924, 1925, ..., 2011
# Theare exactly 90 years, so we also have 90 classes:
nb_classes = 90

data = pd.read_csv('YearPredictionMSD.csv', names=features)

X = data.ix[:,1:].as_matrix()  # this is the 90 columns without the year
Y = data.ix[:,0].as_matrix()   # this is the year column
"""
for i, item in enumerate(X):
    X[i] = (X[i] - np.mean(X[i])) / np.std(X[i])

print(np.mean(X, axis = 0))
print(np.std(X, axis = 0))
print(np.mean(X, axis = 1))
print(np.std(X, axis = 1))
"""
#X = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
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
np.random.seed(147)

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

m_lst = np.array([32, 38, 48, 64, 96, 128, 256])
D_lst = np.array([200, 250, 300, 350, 400, 450])
rank_lst = np.array([15, 30, 40, 50, 60, 70])
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
        #print(np.max(y_pred))
        #print(np.min(y_pred))
        mse_KRR[j,k] = np.mean((y_test_s - y_pred)**2)
        run_time_KRR[j] += time.time() - start_time
    
    
    for j, D in enumerate(D_lst):
        start_time = time.time()
        ran_one = np.zeros(1)
        for l in range(1):
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
            w = np.linalg.solve(ZZ.transpose() @ ZZ + lam * N_train * np.eye(D+1), ZZ.transpose() @ y_train_s) # TODO: check add intercept: solution is this? Probably yes. 
            y_pred = ZZt @ w
            ran_one[l] = np.mean((y_test_s - y_pred)**2)
        print(np.amin(ran_one))
        mse_ran[j,k] = np.amin(ran_one)
        # mse_ran[j,k] = np.mean((y_test_s - y_pred)**2)
        print(mse_ran[j,k])
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

#%%
    
"""
SM = N_train**(-1/2) * Z
new_col = np.ones(N_train).reshape(N_train, 1)
SMM = np.concatenate((SM, new_col), 1)
yh = N_train**(-1/2) * y_train_s
w = np.linalg.solve(SMM.transpose() @ SMM + lam * np.eye(D+1), SMM.transpose() @ yh)
y_pred = ZZt @ w
mse = np.mean((y_test_s - y_pred)**2)
print("Random Feature Approximation 2: ", mse)
#w_l = np.linalg.solve(X_train_s.transpose() @ X_train_s, X_train_s.transpose() @ y_train_s)
"""

K = compute_gram_mat(X_train_s, X_train_s, sigma)
alpha = np.linalg.solve(K + lam * N_train * np.eye(N_train), y_train_s)
K_test = compute_gram_mat(X_test_s, X_train_s, sigma)
y_pred = K_test @ alpha
#print(y_pred)
mse_na = np.mean((y_test_s - y_pred)**2)
print("Kernel Ridge regression: ", mse_na)

new_col = np.ones(N_train).reshape(N_train, 1)
X_train_ss = np.concatenate((X_train_s, new_col), 1)
new_col = np.ones(N_test).reshape(N_test, 1)
X_test_ss = np.concatenate((X_test_s, new_col), 1)
w_l = np.linalg.inv(np.matmul(X_train_ss.transpose(), X_train_ss)) @ X_train_ss.transpose() @ y_train_s
y_pred = np.matmul(X_test_ss, w_l)
mse_lr = np.mean((y_test_s - y_pred)**2)
print("Linear regression: ", mse_lr)

clf = LinearRegression().fit(X_train_s, y_train_s)
y_pred = clf.predict(X_test_s)
#print(y_pred)
lin_err = np.mean((y_test_s - y_pred)**2)
print("Linear Regression sklearn: ", lin_err)

            
#%%

# Kernel Sklearn

gamma = 1/4096
alpha = 2048**(-1)
clf2 = KernelRidge(alpha = alpha, kernel = 'rbf', gamma = gamma)
clf2.fit(X_train_s, y_train_s)
y_pred = clf2.predict(X_test_s)
print("Kernel Ridge sklearn: ", np.mean((y_test_s - y_pred)**2))

#%%

# Cross validation to find best parameter

err_lst = np.zeros((20, 20, 5))
alpha_lst = np.linspace(1/1000, 1/100, num = 20)
#alpha = 1/100
gamma_lst = np.linspace(1/5000, 1/1000, num = 20)
start_time = time.time()
sim_num = 5
for k in range(sim_num):
    ind_1 = [x for x in range(N_train)]
    shuffle(ind_1)
    ind_2 = [x for x in range(N_test)]
    shuffle(ind_2)
    np.take(X_train_s, ind_1, axis = 0, out = X_train_s)
    np.take(X_test_s, ind_2, axis = 0, out = X_test_s)
    np.take(y_train_s, ind_1, axis = 0, out = y_train_s)
    np.take(y_test_s, ind_2, axis = 0, out = y_test_s)
    for i, alpha in enumerate(alpha_lst):
        for j, gamma in enumerate(gamma_lst):
            #clf = Ridge(alpha = alpha).fit(X_train, y_train)
            #y_pred = clf.predict(X_test)
            # print(y_pred)
            #lin_err = np.mean((y_test - y_pred)**2)
            #print("Linear Regression: ", lin_err)
            clf2 = KernelRidge(alpha = alpha, kernel = 'rbf', gamma = gamma)
            cv_results = cross_validate(clf2, X_train_s, y_train_s, cv = 4, scoring=('neg_mean_squared_error'), return_estimator = True)
            print(np.mean(cv_results['test_score']))
            err_lst[i,j,k] = -np.mean(cv_results['test_score'])
            #clf2.fit(X_train, y_train)
            #y_pred = clf2.predict(X_test)
            #err_lst[i] = np.mean((y_test - y_pred)**2)
            print("run time is: ", time.time() - start_time)
            #print("Kernel Ridge Regression: {}, alpha={}, gamma={}".format(ker_err, alpha, gamma))
err_min = np.min(err_lst)
min_ind = np.where(err_lst == err_min)
print(err_lst)
#%%

# Produce Figure 4

ind_1 = [x for x in range(N_train)]
shuffle(ind_1)
ind_2 = [x for x in range(N_test)]
shuffle(ind_2)
np.take(X_train_s, ind_1, axis = 0, out = X_train_s)
np.take(X_test_s, ind_2, axis = 0, out = X_test_s)
np.take(y_train_s, ind_1, axis = 0, out = y_train_s)
np.take(y_test_s, ind_2, axis = 0, out = y_test_s)

m = 1
n = int(N_train/m)
sigma = 32 * np.sqrt(2)
lam = N_train**(-1)
#lam_lst = np.linspace(0.002 * N_train**(-1),0.004 * N_train**(-1), num = 10)
#print(lam_lst)
X_train_split = split_into_m_parts(X_train_s, m) # shape of m, n, d
y_train_split = split_into_m_parts(y_train_s, m)

print(X_train_split.shape)

#for lam in lam_lst:
y_pred_lst = np.zeros((m, N_test))
for i, (XX, yy) in enumerate(zip(X_train_split, y_train_split)):
    K = compute_gram_mat(XX, XX, sigma)
    alpha = np.linalg.solve(K + lam * n * np.eye(n), yy)
    K_test = compute_gram_mat(X_test_s, XX, sigma)
    y_pred_lst[i] = K_test @ alpha
y_pred = np.mean(y_pred_lst, axis = 0)
#print(np.max(y_pred))
#print(np.min(y_pred))
mse = np.mean((y_test_s - y_pred)**2)
print("Fast KRR: ", mse)

X_train_1 = X_train_s[0:n]
y_train_1 = y_train_s[0:n]
K = compute_gram_mat(X_train_1, X_train_1, sigma)
alpha = np.linalg.solve(K + lam * m * n * np.eye(n), yy)
K_test = compute_gram_mat(X_test_s, X_train_1, sigma)
y_pred = K_test @ alpha
#print(y_pred)
mse_na = np.mean((y_test_s - y_pred)**2)
print("Fast KRR no avg: ", mse_na)

"""
# mse = compute_mse(X_train, y_train, N_train, m, params, dist_metric,
        X_test, y_test, real = True)
# mse_na = compute_mse_no_avg(X_train, y_train, N_train, m, params, dist_metric,
        X_test, y_test, real = True)
#clf = KernelRidge().fit(X_train, y_train)
#y_pred = clf.predict(X_test)
#krr_sk = np.mean((y_test - y_pred)**2)

print("Linear Regression: ", lin_err)
#print("Kernel Ridge Regression: ", krr_sk)
"""
#%%


# X_train_std = np.std(X_train, axis = 0)
#X_train = X_train / X_train_std

#X, y = init_sim_data(2000)

#print(np.linalg.norm(X[1] - X[-4]))
#for i, item in enumerate(X_train):
#    X_train[i] = X_train[i] / X_train_std[i]
"""
print(y_train)

print(np.linalg.norm(X_train[0] - X_train[1]))
print(y_train[0])
print(y_train[1])

print(np.linalg.norm(X_train[0] - X_train[2]))
print(y_train[0])
print(y_train[2])

print(np.linalg.norm(X_train[1] - X_train[2]))
print(y_train[1])
print(y_train[2])

print(np.linalg.norm(X_train[0] - X_train[-1]))
print(y_train[0])
print(y_train[-1])

print(np.linalg.norm(X_train[0] - X_train[-2]))
print(y_train[0])
print(y_train[-2])

print(np.linalg.norm(X_train[0] - X_train[-3]))
print(y_train[0])
print(y_train[-2])

print(np.linalg.norm(X_train[1] - X_train[-4]))
print(y_train[0])
print(y_train[-2])

#print(np.std(X_train, axis = 1))

# print(np.linalg.norm(X_train[1] - X_train[-3]))
#print(y_train[1])
#print(y_train[-3])

#print(y_train)
#print(y_train[-3])
"""
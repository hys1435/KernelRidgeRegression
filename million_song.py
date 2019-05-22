#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:49:15 2019

@author: Zhaoqi Li
"""

# The subset of million song dataset is obtained from https://samyzaf.com/ML/song_year/song_year.html

import pandas as pd
from KRR_algorithm import compute_gram_mat, compute_mse, compute_coeffs_from_K, predict
from multiprocessing import Pool
import time

import numpy as np
#%%

# fixed random seed for reproducibility
np.random.seed(0)

features = ['year', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't1 9', 't20', 't21', 't22', 't23', 't24', 't25', 't26', 't27', 't28', 't29', 't30', 't31', 't32', 't33', 't34', 't35', 't36 ', 't37', 't38', 't39', 't40', 't41', 't42', 't43', 't44', 't45', 't46', 't47', 't48', 't49', 't50', 't51', 't52', 't53' , 't54', 't55', 't56', 't57', 't58', 't59', 't60', 't61', 't62', 't63', 't64', 't65', 't66', 't67', 't68', 't69', 't70', 't71', 't72', 't73', 't74', 't75', 't76', 't77', 't78', 't79', 't80', 't81', 't82', 't83', 't84', 't85', 't86', 't87', 't88', 't89', 't90']

# Note that our classes (which we have to predict from those 90 features), are all
# the years from 1922 to 2011: 1922, 1923, 1924, 1925, ..., 2011
# Theare exactly 90 years, so we also have 90 classes:
nb_classes = 90

data = pd.read_csv('YearPredictionMSD.csv', names=features)

X = data.ix[:,1:].as_matrix()  # this is the 90 columns without the year
Y = data.ix[:,0].as_matrix()   # this is the year column

# data normalizations (scaling down all values to the interval [0,1])
# The years 1922-2011 are scaled down to integers [0,1,2,..., 89]
a = X.min()
b = X.max()
X = (X - a) / (b - a)  # all values now between 0 and 1 !
Y = Y - Y.min()        # The years 1922-2011 are mapped to 0-89

# Training data set
X_train = X[0:463715]
y_train = Y[0:463715]

# Validation data set
X_test = X[463715:]
y_test = Y[463715:]


print("------ finished read data ------")
#%%

# Different low-rank approximation methods for the kernel matrix

def Nystrom_sampling(X, rank, params, dist_metric):
    Kmn = compute_gram_mat(X, X[0:rank], params, dist_metric)
    Kmm = compute_gram_mat(X[0:rank], X[0:rank], params, dist_metric)
    Kmm_inv = np.linalg.inv(Kmm)
    return np.matmul(np.matmul(Kmn, Kmm_inv),np.transpose(Kmn))

def rand_fourier_features(X, D, sigma):
    # d: dimension of data; D: dimension of samples
    d = X.shape[1]
    w = np.random.multivariate_normal(mean = np.zeros(d), cov = sigma**2 * np.eye(d), size = D)
    print(w.shape)
    b = np.random.uniform(low = 0, high = 2*np.pi, size = D)
    print(b.shape)
    print(X.shape)
    z = np.sqrt(2/D) * np.cos(np.matmul(w, np.transpose(X)) + b)
    return z

def main():
    start_time = time.time()
    #N = 463715
    #X_train, y_train, X_test, y_test = processData()

    N = 200 # small sample test
    mLst = [4, 8]
    X_train = X[0:N]
    X_test = X[463715:(463715+int(N/10))]
    y_train = Y[0:N]
    y_test = Y[463715:(463715+int(N/10))]

    dist_metric = "gaussian"
    #mLst = [32, 38, 48, 64, 96, 128, 256]
    sigma = 6 * np.sqrt(2) # sqrt(2) is for the version the author uses here: 2*sigma**2
    lam = N**(-1)
    params = [sigma, lam]
    r = 1 * 10**1
    D = 2000
    K_Nys = Nystrom_sampling(X_train, r, params, dist_metric)
    print("K_Nys: ", K_Nys)
    K = compute_gram_mat(X_train, X_train, params, dist_metric)
    print("K: ", K)
    K_ran = rand_fourier_features(X_train, D, sigma)
    print("K_ran: ", K_ran)
    print("Difference: ", np.linalg.norm(K-K_Nys))
    print("Difference: ", np.linalg.norm(K-K_ran))
    
    mse_lst = np.zeros(len(mLst))
    for i, m in enumerate(mLst):
        mse_lst[i] = compute_mse(X_train, y_train, N, m, params, dist_metric,
                    X_test, y_test, real = True)
        #feature_map_nystroem = Nystroem(gamma = 1/sigma**2, n_components = r)
        #K_Nys = feature_map_nystroem.fit_transform(data)
        #alpha = compute_coeffs_from_K(K_Nys, y_train, params)
        #y_pred = predict(X, X_test, alpha, m, params, dist_metric)
        #mse = np.mean((y_test - y_pred)**2)
        print("run time is: ", (time.time() - start_time))
        print(mse_lst[i])
        #print("Nystrom: ", mse)
    print(mse_lst)

if __name__ == '__main__':
     main()


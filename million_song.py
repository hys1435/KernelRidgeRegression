#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:49:43 2019

@author: zli9
"""

# The subset of million song dataset is obtained from https://samyzaf.com/ML/song_year/song_year.html

import pandas as pd
from KRR_algorithm import compute_gram_mat, compute_mse, compute_coeffs_from_K
import time
from random import shuffle
# import matplotlib.pyplot as plt
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

def compute_pseudoinv(K):
    # compute pseudoinverse of K using svd
    U, D, VT = np.linalg.svd(K, full_matrices=False)
    #print(U)
    #print(D)
    #print(VT)
    K_pseudoinv = VT.transpose() @ np.linalg.inv(np.diag(D)) @ U.transpose()
    return K_pseudoinv

def Nystrom_sampling(X, rank, params, dist_metric):
    Kmn = compute_gram_mat(X, X[0:rank], params, dist_metric)
    Kmm = compute_gram_mat(X[0:rank], X[0:rank], params, dist_metric)
    #print("Kmm: ", Kmm)
    Kmm_inv = compute_pseudoinv(Kmm)
    #Kmm_inv = np.linalg.inv(Kmm)
    #print("Kmm_inv: ", Kmm_inv)
    return np.matmul(np.matmul(Kmn, Kmm_inv),np.transpose(Kmn))

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

def main():
    N_train = 463715
    N_test = 51630
    #X_train, y_train, X_test, y_test = processData()

    #N_train = 2000 # small sample test
    #N_test = int(N_train/10)
    #mLst = [32, 38, 48]
    #DLst = [2000, 3000]
    #rLst = [1000, 2000, 3000]
    mLst = np.array([32, 38, 48, 64, 96, 128, 256])
    DLst = np.array([2000, 3000, 5000, 7000, 8500, 10000])
    rLst = np.array([1000, 2000, 3000, 4000, 5000, 6000])
    #X_train = X[0:N_train]
    #X_test = X[463715:(463715+int(N_train/10))]
    #y_train = Y[0:N_train]
    #y_test = Y[463715:(463715+int(N_train/10))]

    dist_metric = "gaussian"
    sigma = 6 * np.sqrt(2) # sqrt(2) is for the version the author uses here: 2*sigma**2
    lam = N_train**(-1)
    params = [sigma, lam]

    sim_num = 10
    run_time_KRR = np.zeros(len(mLst))
    run_time_Nys = np.zeros(len(rLst))
    run_time_ran = np.zeros(len(DLst))
    mse_lst_KRR = np.zeros((len(mLst), sim_num))
    mse_lst_Nys = np.zeros((len(rLst), sim_num))
    mse_lst_ran = np.zeros((len(DLst), sim_num))

    """
    K_Nys = Nystrom_sampling(X_train, r, params, dist_metric)
    #print("K_Nys: ", K_Nys)
    K = compute_gram_mat(X_train, X_train, params, dist_metric)
    #print("K: ", K)
    z_ran = rand_fourier_features(X_train, D, sigma)
    K_ran = np.matmul(z_ran, z_ran.transpose())
    #print("K_ran: ", K_ran)
    #print("Difference: ", np.linalg.norm(K-K_Nys))
    #print("Difference: ", np.linalg.norm(K-K_ran))
    """
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
        for i, m in enumerate(mLst):
            mse_lst_KRR[i,k] = compute_mse(X_train, y_train, N_train, m, params, dist_metric,
                        X_test, y_test, real = True)
            print("mse_KRR: {}, i={}, k={}: ".format(mse_lst_KRR[i,k], i, k))
        run_time_KRR[i] = time.time() - start_time
        """
        start_time = time.time()
        for i, r in enumerate(rLst):
            K_Nys = Nystrom_sampling(X_train, r, params, dist_metric)
            #print("K_Nys: ", K_Nys)
            alpha = compute_coeffs_from_K(K_Nys, y_train, params)
            K = compute_gram_mat(X_test, X_train, params, dist_metric)
            y_pred = np.dot(K, alpha)
            mse_lst_Nys[i,k] = np.mean((y_test - y_pred)**2)
            print("mse_Nys: {}, i={}, k={}: ".format(mse_lst_Nys[i,k], i, k))
        run_time_Nys[i] = time.time() - start_time
        """
        start_time = time.time()
        for i, D in enumerate(DLst):
            z_ran = rand_fourier_features(X_train, D, sigma)
            K_ran = np.matmul(z_ran, z_ran.transpose())
            alpha = compute_coeffs_from_K(K_ran, y_train, params)
            K = compute_gram_mat(X_test, X_train, params, dist_metric)
            y_pred = np.dot(K, alpha)
            mse_lst_ran[i,k] = np.mean((y_test - y_pred)**2)
            print("mse_ran: {}, i={}, k={}: ".format(mse_lst_ran[i,k], i, k))
        run_time_ran[i] = time.time() - start_time
        print("{}th iteration Done".format(k))

    mse_lst_KRR_err = np.std(mse_lst_KRR, axis = 1)
    mse_lst_KRR = np.mean(mse_lst_KRR, axis = 1)
    mse_lst_Nys_err = np.std(mse_lst_Nys, axis = 1)
    mse_lst_Nys = np.mean(mse_lst_Nys, axis = 1)
    mse_lst_ran_err = np.std(mse_lst_ran, axis = 1)
    mse_lst_ran = np.mean(mse_lst_ran, axis = 1)

    np.savetxt("mse_lst_KRR.out", mse_lst_KRR)
    np.savetxt("mse_lst_Nys.out", mse_lst_Nys)
    np.savetxt("mse_lst_ran.out", mse_lst_ran)
    np.savetxt("mse_lst_KRR_err.out", mse_lst_KRR_err)
    np.savetxt("mse_lst_Nys_err.out", mse_lst_Nys_err)
    np.savetxt("mse_lst_ran_err.out", mse_lst_ran_err)
    print("mse_lst_KRR: ", mse_lst_KRR)
    print("mse_lst_Nys: ", mse_lst_Nys)
    print("mse_lst_ran: ", mse_lst_ran)
    print("mse_lst_KRR_err: ", mse_lst_KRR_err)
    print("mse_lst_Nys_err: ", mse_lst_Nys_err)
    print("mse_lst_ran_err: ", mse_lst_ran_err)

"""    
    # Plot results
    fig, ax = plt.subplots()
    cols = ['red', 'blue', 'green']
    markers = ['o', '^', 's']
    ax.errorbar(run_time_KRR, mse_lst_KRR, yerr = mse_lst_KRR_err, c=cols[0], marker=markers[0], label='Fast-KRR')
    ax.errorbar(run_time_Nys, mse_lst_Nys, yerr = mse_lst_Nys_err, c=cols[1], marker=markers[1], label='Nystrom Sampling')
    ax.errorbar(run_time_ran, mse_lst_ran, yerr = mse_lst_ran_err, c=cols[2], marker=markers[2], label='Random Feature Approx.')
    plt.legend(loc='upper right')
    plt.xlabel("Training runtime")
    plt.ylabel("Mean square error")
    #plt.title("Kernel Ridge Regression without under-regularization")

    plt.show()
"""

if __name__ == '__main__':
     main()

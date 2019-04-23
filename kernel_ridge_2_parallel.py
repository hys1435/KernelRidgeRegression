#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:46:02 2019

@author: hys1435
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:23:40 2019

@author: Zhaoqi Li
"""

import numpy as np
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt

def gaussianRBF(u, v, params):
    sigma = params[0]
    if (len((u-v).shape) == 1): 
        return np.exp(-np.einsum('i,i',u-v,u-v)/(sigma**2))
    return np.exp(-np.einsum('ij,ij->i',u-v,u-v)/(sigma**2)) # einsum computes norm squared

def sobolev(u, v, params):
    return 1 + np.minimum(u, v)
    
def get_kernel(dist_metric):
    if (dist_metric == "gaussian"):
        return gaussianRBF
    elif (dist_metric == "sobolev"):
        return sobolev
    
def compute_gram_mat(X1, X2, params, dist_metric):
    kernel = get_kernel(dist_metric)
    gram_mat = np.zeros((X1.shape[0],X2.shape[0]))
    for i, item in enumerate(X2):
        gram_mat[:,i] = kernel(X1, item, params)
    return gram_mat

def compute_kernel_ridge_coeffs(X, y, params, dist_metric):
    lam = params[-1]
    K = compute_gram_mat(X, X, params, dist_metric)
    K_sudinv = np.linalg.inv(K + lam * y.size * np.eye(K.shape[0]))
    alpha = np.dot(K_sudinv, y)
    return alpha

def f_star(x):
    return np.minimum(x, 1-x)

def split_into_m_parts(X, m):
    n = int(X.size / m)
    res = np.zeros((m, n))
    for i in range(m):
        res[i] = X[i*n:(i+1)*n]
    return res

def callbackRes(result):
    print(result)

def predict(X, alpha, m, params, dist_metric, output = False):
    K = compute_gram_mat(X, X, params, dist_metric)
    y_pred = 1/m * np.dot(K, alpha)
    if (output):
        return y_pred, K, alpha
    return y_pred

# Initialize data given N number of samples
def init_data(N):
    X = np.random.uniform(size = N)
    epsilon = np.random.normal(scale = 1/5, size = N)
    y = f_star(X) + epsilon
    return X, y


def main():
    # Initialize global variables
    #np.random.seed(521)
    start_time = time.time()
    NLst = np.logspace(8, 12, num = 6, base = 2).astype(int) # correct one is 8-13
    mLst = np.logspace(0, 3, num = 4, base = 4).astype(int)
    dist_metric = "sobolev"
    mse_lst = np.zeros((mLst.size, NLst.size))
    for i, N in enumerate(NLst):
        X, y = init_data(N)
        lam = N**(-2/3)
        params = [-1, lam] # params are nothing and lambda
        alpha = np.zeros(N)
        for j, m in enumerate(mLst):
            X_split = split_into_m_parts(X, m)
            y_split = split_into_m_parts(y, m)
            n = int(N / m)
            p = Pool(m)
            results = [p.apply_async(compute_kernel_ridge_coeffs, [XX, yy, params, dist_metric]) 
            for XX, yy in zip(X_split, y_split)]
            p.close()
            p.join()
            
            for k, r in enumerate(results):
                alpha[k*n:(k+1)*n] = r.get()
            y_pred = predict(X, alpha, m, params, dist_metric)
            mse_lst[j,i] = np.mean((y - y_pred)**2)
            print("run time is: ", (time.time() - start_time))

    # Plot results
    ax = plt.subplot(111)
    cols = ['red', 'blue', 'yellow', 'orange']
    markers = ['o', '^', 's', 'd']
    plt.xticks(NLst)
    ax.set_yscale('log')
    for i in range(mLst.size):
        ax.plot(NLst, mse_lst[i], c=cols[i], marker=markers[i],label='m={}'.format(4**i))
    ax.legend(loc='upper right')
    
    plt.xlabel("Total number of samples (N)")
    plt.ylabel("Mean square error")
    plt.title("Kernel Ridge Regression without under-regularization")
    
    plt.show()

if __name__ == '__main__':
     main()
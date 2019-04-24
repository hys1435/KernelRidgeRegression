#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:23:40 2019

@author: Zhaoqi Li
"""
import numpy as np
from KRR_algorithm import compute_mse
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt

def f_star(x):
    return np.minimum(x, 1-x)

# Initialize data given N number of samples
def init_sim_data(N):
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
    mse_lst_nr = np.zeros((mLst.size, NLst.size))
    for i, N in enumerate(NLst):
        X, y = init_sim_data(N)
        for j, m in enumerate(mLst):
            lam = N**(-2/3)
            n = int(N / m)
            p = Pool(m)
            params = [-1, lam] # params are nothing and lambda
            lam_nr = n**(-2/3)
            params_nr = [-1, lam_nr]
            mse_lst[j,i] = compute_mse(X, y, N, m, p, params, dist_metric)
            mse_lst_nr[j,i] = compute_mse(X, y, N, m, p, params_nr, dist_metric)
            p.close()
            p.join()
            print("run time is: ", (time.time() - start_time))

    # Plot results
    ax = plt.subplot(2,1,1)
    cols = ['red', 'blue', 'yellow', 'orange']
    markers = ['o', '^', 's', 'd']
    plt.xticks(NLst)
    ax.set_yscale('log')
    for i in range(mLst.size):
        ax.plot(NLst, mse_lst[i], c=cols[i], marker=markers[i],label='m={}'.format(4**i))
    plt.legend(loc='upper right')
    
    ax2 = plt.subplot(2,1,2)
    for i in range(mLst.size):
        ax2.plot(NLst, mse_lst_nr[i], c=cols[i], marker=markers[i],label='m={}'.format(4**i))
    plt.legend(loc='upper right')
    plt.xlabel("Total number of samples (N)")
    plt.ylabel("Mean square error")
    plt.title("Kernel Ridge Regression without under-regularization")
    
    
    plt.show()

if __name__ == '__main__':
     main()
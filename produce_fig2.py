#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:51:43 2019

@author: zli9
"""

import numpy as np
from KRR_algorithm import compute_mse
from sim_study_helper_funs import init_sim_data, init_params
import time
import matplotlib.pyplot as plt
    
def main():
    # Initialize global variables
    #np.random.seed(521)
    start_time = time.time()
    NLst = np.logspace(8, 12, num = 6, base = 2).astype(int) # correct one is 8-13
    dist_metric = "sobolev"
    mse_lst = list()
    for i, N in enumerate(NLst):
        X, y = init_sim_data(N)
        logmnLst = np.linspace(1/N, 1)
        mse = np.zeros(logmnLst.size)
        mLst = np.logspace(1, N).astype(int)
        for j, m in enumerate(mLst):
            lam, n, p, params = init_params(N, m)
            mse[j] = compute_mse(X, y, N, m, p, params, dist_metric)
            p.close()
            p.join()
            print("run time is: ", (time.time() - start_time))
        mse_lst.append(mse)
    """
    # Plot results
    ax = plt.subplot(3,1,1)
    cols = ['red', 'blue', 'yellow', 'orange']
    markers = ['o', '^', 's', 'd']
    plt.xticks(NLst)
    ax.set_yscale('log')
    for i in range(mLst.size):
        ax.plot(NLst, mse_lst[i], c=cols[i], marker=markers[i],label='m={}'.format(4**i))
    plt.legend(loc='upper right')
    
    ax2 = plt.subplot(3,1,2)
    for i in range(mLst.size):
        ax2.plot(NLst, mse_lst_nr[i], c=cols[i], marker=markers[i],label='m={}'.format(4**i))
    plt.legend(loc='upper right')
    plt.xlabel("Total number of samples (N)")
    plt.ylabel("Mean square error")
    plt.title("Kernel Ridge Regression without under-regularization")

    plt.show()
    """
if __name__ == '__main__':
     main()
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
    NLst = np.logspace(8, 13, num = 6, base = 2).astype(int) # correct one is 8-13
    sim_num = 20
    dist_metric = "sobolev"
    mse_lst = list()
    logmnLst = list()
    for i, N in enumerate(NLst):
        logN = np.log2(N)
        logmn = np.linspace(0, 1, num = logN, endpoint = False)
        mse = np.zeros([logmn.size, sim_num])
        mLst = np.logspace(start = 0, stop = logN, num = logN, base = 2, endpoint = False).astype(int)
        for k in range(sim_num):
            X, y = init_sim_data(N)
            for j, m in enumerate(mLst):
                lam, n, params = init_params(N, m)
                mse[j, k] = compute_mse(X, y, N, m, params, dist_metric)
        print("run time is: ", (time.time() - start_time))
        mse_mean = np.mean(mse, axis = 1)
        mse_lst.append(mse_mean)
        logmnLst.append(logmn)
    print(mse_lst)

    # Plot results
    fig, ax = plt.subplots()
    cols = ['red', 'blue', 'yellow', 'orange', 'green', 'purple']
    markers = ['o', '^', 's', 'd', '*', 'P']
    ax.set_yscale('log')
    for i, (logmn, mse) in enumerate(zip(mse_lst, logmnLst)):
        ax.plot(mse, logmn, c=cols[i], marker=markers[i],label='N={}'.format(2**(i+8)))
    ax.set_yticks(np.array([0.0003, 0.001, 0.005, 0.01, 0.02]))
    plt.legend(loc='lower right')
    plt.xlabel("log(# of partitions)/log(# of samples)")
    plt.ylabel("Mean square error")
    
    plt.show()

if __name__ == '__main__':
     main()
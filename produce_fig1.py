#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:23:40 2019

@author: Zhaoqi Li
"""
import numpy as np
from KRR_algorithm import compute_mse
from sim_study_helper_funs import init_sim_data, init_params
import time
import matplotlib.pyplot as plt
    
# code to reproduce Figure 1 of zhang15d paper

def main():
    # Initialize global variables
    #np.random.seed(521)
    start_time = time.time()
    NLst = np.logspace(8, 12, num = 5, base = 2).astype(int) # correct one is 8-13
    mLst = np.logspace(0, 3, num = 4, base = 4).astype(int)
    dist_metric = "sobolev"
    sim_num = 20
    mse_lst = np.zeros((mLst.size, NLst.size, sim_num)) # list of mse with under-regularization
    mse_lst_nr = np.zeros((mLst.size, NLst.size, sim_num)) # list of mse without under-regularization
    for k in range(sim_num):
        for i, N in enumerate(NLst):
            X, y = init_sim_data(N)
            for j, m in enumerate(mLst):
                lam, n, params = init_params(N, m)
                lam_nr = n**(-2/3)
                params_nr = [-1, lam_nr]
                mse_lst[j,i,k] = compute_mse(X, y, N, m, params, dist_metric)
                mse_lst_nr[j,i,k] = compute_mse(X, y, N, m, params_nr, dist_metric)
            #p.close()
            #p.join()
        print("run time is: ", (time.time() - start_time))

    mse_lst = np.std(mse_lst, axis = 2)
    mse_lst_nr = np.std(mse_lst_nr, axis = 2)
    print(mse_lst_nr)
    
    # Plot results
    ax = plt.subplot(2,1,1)
    cols = ['red', 'blue', 'yellow', 'orange']
    markers = ['o', '^', 's', 'd']
    plt.xticks(NLst)
    plt.yscale('log')
    for i in range(mLst.size):
        ax.plot(NLst, mse_lst[i], c=cols[i], marker=markers[i],label='m={}'.format(4**i))
        #ax.set_yscale('log')
    plt.legend(loc='upper right')
    
    ax2 = plt.subplot(2,1,2)
    plt.xticks(NLst)
    for i in range(mLst.size):
        ax2.plot(NLst, mse_lst_nr[i], c=cols[i], marker=markers[i],label='m={}'.format(4**i))
        ax2.set_yscale('log')
    plt.legend(loc='upper right')
    plt.xlabel("Total number of samples (N)")
    plt.ylabel("Mean square error")
    plt.title("Kernel Ridge Regression without under-regularization")

    plt.show()

if __name__ == '__main__':
     main()
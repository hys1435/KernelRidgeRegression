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
import matplotlib.ticker
from random import shuffle
    
# code to reproduce Figure 1 of zhang15d paper

def main():
    # Initialize global variables
    #np.random.seed(521)
    start_time = time.time()
    NLst = np.logspace(8, 12, num = 5, base = 2).astype(int) # correct one is 8-13
    mLst = np.logspace(0, 3, num = 4, base = 4).astype(int)
    dist_metric = "sobolev"
    sim_num = 10
    mse_lst = np.zeros((mLst.size, NLst.size, sim_num)) # list of mse with under-regularization
    mse_lst_nr = np.zeros((mLst.size, NLst.size, sim_num)) # list of mse without under-regularization
    for k in range(sim_num):
        for i, N in enumerate(NLst):
            X, y = init_sim_data(N)
            ind = [x for x in range(N)]
            shuffle(ind)
            np.take(X, ind, axis = 0, out = X)
            np.take(y, ind, axis = 0, out = y)
            for j, m in enumerate(mLst):
                lam, n, params = init_params(N, m)
                lam_nr = n**(-2/3)
                params_nr = [1, lam_nr]            
                mse_lst[j,i,k] = compute_mse(X, y, N, m, params, dist_metric)
                mse_lst_nr[j,i,k] = compute_mse(X, y, N, m, params_nr, dist_metric)
        print("run time is: ", (time.time() - start_time))

    mse_lst = np.mean(mse_lst, axis = 2)
    mse_lst_nr = np.mean(mse_lst_nr, axis = 2)
    #print(mse_lst)
    #print(mse_lst_nr)
    
    # Plot results
    fig, ax = plt.subplots()
    cols = ['red', 'blue', 'yellow', 'orange']
    markers = ['o', '^', 's', 'd']
    ax.set_xscale('log')
    plt.yscale('log')
    plt.xticks(NLst)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    for i in range(mLst.size):
        ax.plot(NLst, mse_lst[i], c=cols[i], marker=markers[i],label='m={}'.format(4**i))
    ax.set_yticks(np.array([0.0005, 0.001, 0.0025, 0.005, 0.01, 0.05]))
    plt.legend(loc='upper right')
    plt.xlabel("Total number of samples (N)")
    plt.ylabel("Mean square error")
    plt.show()
    
    fig2, ax2 = plt.subplots()
    ax2.set_xscale('log')
    plt.yscale('log')
    plt.xticks(NLst)
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    for i in range(mLst.size):
        ax2.plot(NLst, mse_lst_nr[i], c=cols[i], marker=markers[i],label='m={}'.format(4**i))
    ax2.set_yticks(np.array([0.0005, 0.001, 0.0025, 0.005, 0.01, 0.05]))
    plt.legend(loc='upper right')
    plt.xlabel("Total number of samples (N)")
    plt.ylabel("Mean square error")

    plt.show()

if __name__ == '__main__':
     main()
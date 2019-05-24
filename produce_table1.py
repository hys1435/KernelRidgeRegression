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
    
# code to reproduce Table 1 of zhang15d paper

def main():
    NLst = np.array([2**12, 2**13, 2**14, 2**15, 2**16, 2**17])
    mLst = np.array([1, 16, 64, 256, 1024])
    dist_metric = "sobolev"
    sim_num = 10
    mse_lst = np.zeros((NLst.size, mLst.size, sim_num))
    run_time_lst = np.zeros((NLst.size, mLst.size, sim_num))
    for k in range(sim_num):
        for i, N in enumerate(NLst):
            X, y = init_sim_data(N)
            for j, m in enumerate(mLst):
                if (m == 1 and N > 2**14+1):
                    print("skip")
                else:
                    start_time = time.time()
                    lam, n, params = init_params(N, m)
                    mse_lst[i,j,k] = compute_mse(X, y, N, m, params, dist_metric)
                    run_time_lst[i,j,k] = time.time() - start_time
        print("run time is: ", (time.time() - start_time))

    mse_lst = np.mean(mse_lst, axis = 2)
    run_time_lst_sd = np.std(run_time_lst, axis = 2)
    run_time_lst = np.mean(run_time_lst, axis = 2)
    print("Error List: ", mse_lst)
    print("Running time list: ", run_time_lst)
    print("Running time sd list: ", run_time_lst_sd)

if __name__ == '__main__':
     main()
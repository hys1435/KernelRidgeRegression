import pandas as pd
from KRR_algorithm import compute_gram_mat, compute_mse, compute_coeffs_from_K, predict
from multiprocessing import Pool
import time

def main():
	start_time = time.time()
	N = 463715
    mLst = [32, 38, 48, 64, 96, 128, 256]
    dist_metric = "gaussian"
    sim_num = 20
    mse_lst = np.zeros((mLst.size, sim_num)) # list of mse with under-regularization
    mse_lst_na = np.zeros((mLst.size, sim_num)) # list of mse with under-regularization
	for k in range(sim_num):
        for j, m in enumerate(mLst):
            lam, n, params = init_params(N, m)
            mse_lst[j,k] = compute_mse(X, y, N, m, params, dist_metric)
            mse_lst_na[j,k] = compute_mse_no_avg(X, y, N, m, params, dist_metric)
        print("run time is: ", (time.time() - start_time))

    mse_lst = np.mean(mse_lst, axis = 1)
    mse_lst_na = np.mean(mse_lst_na, axis = 1)
    print(mse_lst)
    print(mse_lst_na)
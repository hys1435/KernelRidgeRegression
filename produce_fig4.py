import numpy as np
from KRR_algorithm import compute_mse, compute_mse_no_avg
from sim_study_helper_funs import init_params
from process_data import processData
import time
import matplotlib.pyplot as plt

def main():
    start_time = time.time()
    N = 463715
    mLst = np.array([32, 38, 48, 64, 96, 128, 256])
    
    X_train, y_train, X_test, y_test = processData()
    N = 512 # small sample test
    mLst = np.array([8, 32])
    X_train = X_train[0:N]
    X_test = X_test[0:N]
    y_train = y_train[0:N]
    y_test = y_test[0:N]
    
    dist_metric = "gaussian"
    sim_num = 20
    mse_lst = np.zeros((mLst.size, sim_num)) # list of mse with under-regularization
    mse_lst_na = np.zeros((mLst.size, sim_num)) # list of mse with under-regularization
    for k in range(sim_num):
        for j, m in enumerate(mLst):
            lam, n, params = init_params(N, m)
            mse_lst[j,k] = compute_mse(X_train, y_train, N, m, params, dist_metric,
                    X_test, y_test, real = True)
            mse_lst_na[j,k] = compute_mse_no_avg(X_train, y_train, N, m, params, dist_metric,
                    X_test, y_test, real = True)
        print("run time is: ", (time.time() - start_time))

    mse_lst_err = np.std(mse_lst, axis = 1)
    mse_lst = np.mean(mse_lst, axis = 1)
    mse_lst_na_err = np.std(mse_lst_na, axis = 1)
    mse_lst_na = np.mean(mse_lst_na, axis = 1)
    
    print(mse_lst)
    print(mse_lst_err)
    print(mse_lst_na)
    print(mse_lst_na_err)
    
    # Plot results
    fig, ax = plt.subplots()
    cols = ['red', 'blue']
    markers = ['o', 's']
    ax.set_yscale('log')
    ax.errorbar(mLst, mse_lst, yerr = mse_lst_err, c=cols[0], marker=markers[0],label='Fast-KRR')
    ax.errorbar(mLst, mse_lst_na, yerr = mse_lst_na_err, c=cols[1], marker=markers[1],label='KRR with 1/m data')
    plt.legend(loc='upper left')
    plt.xlabel("log(# of partitions)/log(# of samples)")
    plt.ylabel("Mean square error")

if __name__ == '__main__':
     main()
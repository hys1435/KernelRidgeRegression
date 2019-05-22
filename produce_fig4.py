import numpy as np
from KRR_algorithm import compute_mse, compute_mse_no_avg
from sim_study_helper_funs import init_params
from process_data import processData
from random import shuffle
import time
import matplotlib.pyplot as plt

def main():
    start_time = time.time()
    N_train = 463715
    N_test = 51630
    
    mLst = np.array([32, 38, 48, 64, 96, 128, 256])
    
    X_train, y_train, X_test, y_test = processData()
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    mLst = np.array([32, 38])
    """
    N = 512 # small sample test
    X_train = np.array(X_train[0:N])
    X_test = np.array(X_test[0:N])
    y_train = np.array(y_train[0:N])
    y_test = np.array(y_test[0:N])
    """
    
    dist_metric = "gaussian"
    sim_num = 2
    mse_lst = np.zeros((mLst.size, sim_num)) # list of mse with under-regularization
    mse_lst_na = np.zeros((mLst.size, sim_num)) # list of mse with under-regularization
    for k in range(sim_num):
        ind_1 = [x for x in range(N_train)]
        shuffle(ind_1)
        ind_2 = [x for x in range(N_test)]
        shuffle(ind_2)
        np.take(X_train, ind_1, axis = 0, out = X_train)
        np.take(X_test, ind_2, axis = 0, out = X_test)
        np.take(y_train, ind_1, axis = 0, out = y_train)
        np.take(y_test, ind_2, axis = 0, out = y_test)
        for j, m in enumerate(mLst):
            lam, n, params = init_params(N_train, m)
            mse_lst[j,k] = compute_mse(X_train, y_train, N_train, m, params, dist_metric,
                    X_test, y_test, real = True)
            mse_lst_na[j,k] = compute_mse_no_avg(X_train, y_train, N_train, m, params, dist_metric,
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
import numpy as np
from multiprocessing import Pool
import time

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
    return K, alpha

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

def main():
    N = 2**13
    m = 4
    p = Pool(m)
    start_time = time.time()
    lam = N**(-2/3)
    params = [-1, lam] # params are nothing and lambda
    dist_metric = "sobolev"
    
    X = np.random.uniform(size = N)
    epsilon = np.random.normal(scale = 1/5, size = N)
    y = f_star(X) + epsilon
    X_split = split_into_m_parts(X, m)
    eps_split = split_into_m_parts(epsilon, m)
    y_split = split_into_m_parts(y, m)
    """
    # for loop without parallelizing
    for X, y in zip(X_split, y_split):
        K, alpha = compute_kernel_ridge_coeffs(X, y, params, dist_metric)
    """
    # for loop with parallelizing
    results = [p.apply_async(compute_kernel_ridge_coeffs, [X, y, params, dist_metric], 
                             callback = callbackRes) for X, y in zip(X_split, y_split)]
    p.close()
    p.join()
    
    print("run time is: ", (time.time() - start_time))

if __name__ == '__main__':
     main()
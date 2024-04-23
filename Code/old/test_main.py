from Gaussian_generate import *
from LP_OptCoupling import *
from SC_OptTuple import *
from Interp_QP import *
import gurobipy as gp
import numpy as np
import time
import matplotlib.pyplot as plt
import json

# Record the start time
start_time = time.time()

K = 5  # Number of Gaussian distributions
dim = 3  # Dimensionality of the Gaussian distributions
nu = generate_gaussians(1, dim, seed=38)
mu = generate_gaussians(1, dim, seed=38)
n_X = 10  # Number of samples from the first Gaussian distribution
n_Y = 10  # Number of samples from the second Gaussian distribution
#  # Number of random vectors to generate

print(mu[0])
print(nu[0])
breakpoint()

BY = generate_gaussian_vectors(nu[0], n_Y, seed=38)
BX = generate_gaussian_vectors(mu[0], n_X, seed=38)

breakpoint()

# Compute the optimal coupling matrix
# hat_pi_star = solve_OptCoupling_matrix(BX, BY)

# Compute the optimal tuple
lambda_lower = 0.1
lambda_upper = 5
# Opt_BIg, Opt_varphi = solve_qcqp(BX, BY, hat_pi_star, lambda_lower, lambda_upper)

# # breakpoint()   
# # Define the notations
# tilde_BG, V = notation_star(Opt_BIg, Opt_varphi, BX, lambda_lower, lambda_upper)

# Tau = 100 # Number of iterations
# value_list = []
# map_list = []
# beta = 0.1  #smoothing parameter
# mean_eta = np.zeros(dim)
# covariance_matrix_eta = beta * np.eye(dim)
# np.random.seed(250)
# x = np.random.randint(10, size = dim)

def KS_SCLS_construct(BX, BY, lambda_lower, lambda_upper, iter):
    # Compute the optimal coupling matrix
    hat_pi_star = solve_OptCoupling_matrix(BX, BY)

    # Compute the optimal tuple
    Opt_BIg, Opt_varphi = solve_qcqp(BX, BY, hat_pi_star, lambda_lower, lambda_upper)

    # notations
    tilde_BG, V = notation_star(Opt_BIg, Opt_varphi, BX, lambda_lower, lambda_upper)

    output_file = "output_{}.json".format(iter)
    data = {
        "BX": BX.tolist(),
        "BY": BY.tolist(),
        "lambda_lower": lambda_lower,
        "lambda_upper": lambda_upper,
        "tilde_BG": tilde_BG.toarray().tolist(),
        "V": V.tolist()
    }
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)  # Serialize data to JSON with indentation for readability

    # return tilde_BG, V

# Compute the KS-SCLS
# INPUT:
# x - random vector;
# BX - samples from the mu;
# lambda_lower - strong convexity parameter;
# lambda_upper - smoothness parameter;
# tilde_BG;
# V;
# Tau - number of iterations;
# beta - smoothing parameter.
# OUTPUT:
# hat_varphi_beta;
# hat_T_beta.
        

def read_output(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.asarray(data["BX"]), \
            data["lambda_lower"], \
            data["lambda_upper"], \
            np.asarray(data["tilde_BG"]), \
            np.asarray(data["V"])

iter = 0
KS_SCLS_construct(BX, BY, lambda_lower, lambda_upper, iter)
file_path = "output_{}.json".format(iter)
BX, lambda_lower, lambda_upper, tilde_BG, V = read_output(file_path)

def KS_SCLS_compute(x, BX, lambda_lower, lambda_upper, tilde_BG, V, Tau = 100, beta = 0.1):
    dim = len(x)
    value_list = []
    map_list = []

    mean_eta = np.zeros(dim)
    covariance_matrix_eta = beta * np.eye(dim)

    count = 0
    for t in range(Tau):
        np.random.seed(t)
        eta = np.random.multivariate_normal(mean_eta, covariance_matrix_eta)
        tilde_BIg_star, tilde_varphi_star = Interp_QP((x + eta).reshape(-1, 1), BX, tilde_BG, V, lambda_lower, lambda_upper)
        value_list.append(tilde_varphi_star)
        map_list.append(tilde_BIg_star)
        count += 1

    tilde_varphi_beta = sum(value_list) / Tau
    tilde_BIg_beta = np.sum(map_list, axis=0) / Tau

    hat_varphi_beta = tilde_varphi_beta + (lambda_lower/2) * np.linalg.norm(x)**2
    hat_T_beta = tilde_BIg_beta + lambda_lower * x

    return hat_varphi_beta, hat_T_beta.reshape(-1, 1)

# count = 0
# for t in range(Tau):
#     np.random.seed(t)
#     eta = np.random.multivariate_normal(mean_eta, covariance_matrix_eta)
#     # breakpoint()
#     tilde_BIg_star, tilde_varphi_star = Interp_QP((x + eta).reshape(-1, 1), BX, tilde_BG, V, lambda_lower, lambda_upper)
#     value_list.append(tilde_varphi_star)
#     map_list.append(tilde_BIg_star)
#     count += 1

# print("count", count)  

# tilde_varphi_beta = sum(value_list) / Tau
# tilde_BIg_beta = np.sum(map_list, axis=0) / Tau

# # Plotting
# # plt.figure(figsize=(8, 6))
# # plt.plot(np.arange(1, Tau + 1), value_list, marker='o', linestyle='-', color='b')
# # plt.axhline(y=tilde_varphi_beta, color='r', linestyle='--', label='Average Value')
# # plt.title('Values vs. Iterations')
# # plt.xlabel('Iteration')
# # plt.ylabel('Value')
# # plt.grid(True)
# # plt.show()

# hat_varphi_beta = tilde_varphi_beta + (lambda_lower/2) * np.linalg.norm(x)**2
# hat_T_beta = tilde_BIg_beta + lambda_lower * x
x = np.random.randint(10, size = dim)
hat_varphi_beta, hat_T_beta = KS_SCLS_compute(x, BX, lambda_lower, lambda_upper, tilde_BG, V, Tau = 100, beta = 0.1)

print(f"Optimal value: {hat_varphi_beta}")
print(f"Optimal solution: {hat_T_beta}")
print("type", type(hat_T_beta))

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time:.2f} seconds")



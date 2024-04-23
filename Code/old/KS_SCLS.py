from Gaussian_generate import *
from LP_OptCoupling import *
from SC_OptTuple import *
from Interp_QP import *
import gurobipy as gp
import numpy as np
import time
import matplotlib.pyplot as plt
import json

# Constructing the KS-SCLS
# INPUT:
# BX - samples from the mu;
# BY - samples from the nu;
# lambda_lower - strong convexity parameter;
# lambda_upper - smoothness parameter.
# OUTPUT:
# tilde_BG
# V

def KS_SCLS_construct(BX, BY, lambda_lower, lambda_upper):
    # Compute the optimal coupling matrix
    hat_pi_star, _ = solve_OptCoupling_matrix(BX, BY)

    # Compute the optimal tuple
    Opt_BIg, Opt_varphi = solve_qcqp(BX, BY, hat_pi_star, lambda_lower, lambda_upper)

    # notations
    tilde_BG, V = notation_star(Opt_BIg, Opt_varphi, BX, lambda_lower, lambda_upper)

    # breakpoint()

    return tilde_BG, V

    # output_file = "output_{}.json".format(iter)
    # data = {
    #     "BX": BX.tolist(),
    #     "BY": BY.tolist(),
    #     "lambda_lower": lambda_lower,
    #     "lambda_upper": lambda_upper,
    #     "tilde_BG": tilde_BG.toarray().tolist(),
    #     "V": V.tolist()
    # }
    # with open(output_file, 'w') as f:
    #     json.dump(data, f, indent=4)  # Serialize data to JSON with indentation for readability


# Read the output file
# INPUT:
# file_path - path to the output file.
# OUTPUT:
# BX;
# lambda_lower;
# lambda_upper;
# tilde_BG;
# V.
        
def read_output(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.asarray(data["BX"]), \
            data["lambda_lower"], \
            data["lambda_upper"], \
            np.asarray(data["tilde_BG"]), \
            np.asarray(data["V"])


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

print(111)

def KS_SCLS_compute(x, BX, lambda_lower, lambda_upper, tilde_BG, V, Tau = 100, beta = 0.1):
    dim = len(x)
    value_list = []
    map_list = []

    mean_eta = np.zeros(dim)
    covariance_matrix_eta = beta * np.eye(dim)

    count = 0
    for t in range(Tau):
        np.random.seed(t)
        eta = np.random.multivariate_normal(mean_eta, covariance_matrix_eta).reshape(-1, 1)
        tilde_BIg_star, tilde_varphi_star = Interp_QP((x + eta), BX, tilde_BG, V, lambda_lower, lambda_upper)
        value_list.append(tilde_varphi_star)
        map_list.append(tilde_BIg_star)
        count += 1

    tilde_varphi_beta = sum(value_list) / Tau
    tilde_BIg_beta = np.sum(map_list, axis=0) / Tau 

    hat_varphi_beta = tilde_varphi_beta + (lambda_lower/2) * np.linalg.norm(x)**2
    hat_T_beta = tilde_BIg_beta.reshape(-1, 1) + lambda_lower * x

    # breakpoint() 

    return hat_varphi_beta, hat_T_beta


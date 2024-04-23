import scipy as sp
from scipy.linalg import sqrtm, pinv, norm
import numpy as np
import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory by appending ".." to the current directory
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Add the parent directory to the Python module search path
sys.path.append(parent_dir)

from Iterative_Scheme import * 
from FP_method import *



measure = Measure()
mix_number = 3
gaussians = measure.generate_random_parameters(mix_number, 2, seed = 100)

info = {}
info['Number of mixing gaussians'] = mix_number
info['Information of the mixing gaussians'] = [(arr1.tolist(), arr2.tolist()) for arr1, arr2 in gaussians]
save_data(data = info, pathname = "mixture_input/FP_data", filename = "info.json")

nu_0 = Measure("mixture_gaussian", gaussians)
truncated_sample = nu_0.generate_truncated_sample(2000, 1, A = np.eye(2), b = np.zeros(2))
truncated_mean = np.mean(truncated_sample, axis = 0)
truncated_cov = (truncated_sample - truncated_mean).T @ (truncated_sample - truncated_mean) / 1999

dim = 2
K = 4
nu_paras = nu_0.generate_random_parameters(K, dim = 2, seed = 200)
mean_list = []
cov_list = []

truncated_info = {}
truncated_info["Truncated mean of the base measure"] = truncated_mean.tolist()
truncated_info["Truncated covariance matrix of the base measure"] = truncated_cov.tolist()
for i in range(K):
    mean = truncated_mean + nu_paras[i][0]
    cov = nu_paras[i][1] @ truncated_cov @ nu_paras[i][1].T
    mean_list.append(mean)
    cov_list.append(cov)
    truncated_info['Mean of truncated nu_{}'.format(i)] = mean.tolist()
    truncated_info['Covariance matrix of truncated nu_{}'.format(i)] = cov.tolist()
save_data(data = truncated_info, pathname = "mixture_input/FP_data", filename = "truncated_info.json")

FP = FP_method(dim, mean_list, cov_list)
mu = FP.compute_bary_mean()
Sigma = np.eye(dim)
V_Sigma = FP.compute_V(Sigma)
V_list = [V_Sigma]
difference = math.inf
while difference > 1e-5:
    Sigma = FP.compute_bary_cov(Sigma)
    V_Sigma = FP.compute_V(Sigma)
    difference = abs(V_Sigma - V_list[-1])
    V_list.append(V_Sigma)

solve_data = {}
solve_data['Barycenter mean'] = mu.tolist()
solve_data['Barycenter covariance'] = Sigma.tolist()
solve_data['Objective value'] = V_Sigma
solve_data['Objective value list'] = V_list
save_data(data = solve_data, pathname = "mixture_input/FP_data", filename = "solve_data.json")
print(mu)
print(Sigma)
print(V_Sigma)
print(V_list)

n_samples = 50
mu_barycenter = mu
cov_barycenter = Sigma
BX = nu_0.generate_truncated_sample(50, 1, A = pinv(sqrtm(truncated_cov)) @ sqrtm(cov_barycenter), b = mu_barycenter  - truncated_mean)
# barycenter = Measure("gaussian", [(mu, Sigma)])
# BX = barycenter.generate_truncated_sample(50, 1)
V_value = 0
for i in range(K):
    BY = nu_0.generate_truncated_sample(50, 1, A = cov_list[i], b = mean_list[i])
    W2_square = Iterative_Scheme(dim, K, n_samples).W2_square(BX, BY)
    V_value += W2_square
V_value = V_value / K   
print(V_value)


    



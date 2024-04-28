import scipy as sp
from scipy.linalg import sqrtm, pinv, norm
import numpy as np
import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from Iterative_Scheme import * 
from FP_method import *

dim = 2
K = 4
measure = Measure(dim, "gaussian", [])
base_mu = np.zeros(dim)
base_sample = measure.generate_truncated_sample(size = 2000, R = 1, A = np.eye(dim), b = np.zeros(dim))
base_cov = (base_sample - base_mu).T @ (base_sample - base_mu) / 1999
base_A = sqrtm(base_cov)
# base_sample = pinv(base_A) @ (base_sample - base_mu)

nu_para = measure.generate_random_parameters(K, seed = 41)
dict_info = {}
dict_info['Dimension'] = dim
dict_info['Number of measures'] = K
dict_info["Generating seed"] = 41
dict_info['Base mean'] = base_mu.tolist()
dict_info['Base covariance'] = base_cov.tolist()
dict_info['Base A matrix'] = base_A.tolist()

for i in range(K):
    mean, A = nu_para[i]
    dict_info['Mean of nu_{}'.format(i)] = mean.tolist()
    dict_info['A_matrix of nu_{}'.format(i)] = A.tolist()

save_data(data = dict_info, pathname = "gaussian_input/FP_data", filename = "info.json")

# breakpoint()

nu_list = []
for i in range(K):
    nu_list.append(Measure(dim, "gaussian", [nu_para[i]]))

truncated_measures = {}
truncated_mu_list = []
truncated_cov_list = []
truncated_A_list = []

for i in range(K):
    truncated_mu = nu_para[i][0]
    truncated_cov = nu_para[i][1] @ nu_para[i][1].T
    # truncated_sample = nu_list[i].generate_truncated_sample(size = 2000, R = 1, A = nu_para[i][1]@ pinv(base_A) , b = nu_para[i][0])
    # truncated_cov = (truncated_sample - truncated_mu).T @ (truncated_sample - truncated_mu) / 1999
    truncated_measures['Mean of truncated nu_{}'.format(i)] = truncated_mu.tolist()
    truncated_measures['Covariance matrix of truncated nu_{}'.format(i)] = truncated_cov.tolist()
    truncated_mu_list.append(truncated_mu)
    truncated_cov_list.append(truncated_cov)

save_data(data = truncated_measures, pathname = "gaussian_input/FP_data", filename = "truncated_info.json")

FP = FP_method(dim, truncated_mu_list, truncated_cov_list)
mu, mu_cost = FP.compute_bary_mean()

# breakpoint()

Sigma = np.eye(dim)
V_Sigma = FP.compute_V(Sigma) + mu_cost
V_list = [V_Sigma]
difference = math.inf
while difference > 1e-5:
    Sigma = FP.compute_bary_cov(Sigma)
    V_Sigma = FP.compute_V(Sigma) + mu_cost
    difference = abs(V_Sigma - V_list[-1])
    V_list.append(V_Sigma)

solve_data = {}
solve_data['Barycenter mean'] = mu.tolist()
solve_data['Barycenter covariance'] = Sigma.tolist()
solve_data['Objective value'] = V_Sigma
solve_data['Objective value list'] = V_list
save_data(data = solve_data, pathname = "gaussian_input/FP_data", filename = "solve_data.json")

print(mu)
print(Sigma)
print(V_Sigma)
print(V_list)

# breakpoint()

n_samples = 100
barycenter = Measure(dim, "gaussian", [(mu, Sigma)])
A_bary = sqrtm(Sigma)
b_bary = mu
# sqrtm_Sigma = sqrtm(Sigma)
# A_bary = sqrtm_Sigma @ pinv(base_A)
# b_bary = mu
barycenter_info = {}
barycenter_info['b of barycenter'] = mu.tolist()
barycenter_info['A of barycenter'] = A_bary.tolist()
barycenter_info['Sigma of barycenter'] = Sigma.tolist()
save_data(data = barycenter_info, pathname = "gaussian_input/FP_data", filename = "barycenter_info.json")

BX = barycenter.generate_truncated_sample(size =200, R = 1, A = A_bary @ pinv(base_A), b = b_bary)
save_data(data = BX.tolist(), pathname = "gaussian_input/FP_data", filename = "barycenter_sample.json")
V_value = 0
for i in range(K):
    BY = nu_list[i].generate_truncated_sample(size = 200, R = 1, A = nu_para[i][1] @ pinv(base_A), b = nu_para[i][0])
    W2_square = Iterative_Scheme(dim, K, n_samples).W2_square(BX, BY)
    V_value += W2_square
V_value = V_value / K   
print(V_value)

bary_objective = V_value
save_data(data = bary_objective, pathname = "gaussian_input/FP_data", filename = "bary_objective.json")
    







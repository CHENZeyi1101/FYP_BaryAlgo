import numpy as np
from Gaussian_generate import *
from KS_SCLS import *
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Compute the objective value of V() for a given x
# INPUT:
# x - m-dimensional vector (numpy array)
# input_measures - list of tuples, where each tuple contains the mean vector and covariance matrix of a Gaussian distribution;
# OUTPUT:
# objective - objective value of V() for the given x.

def objective_compute(X, input_measures, n_samples):
    K = len(input_measures)
    objective = 0
    for k in range(K):
        Y = generate_gaussian_vectors(input_measures[k], n_samples, seed=None)
        _, opt_value = solve_OptCoupling_matrix(X, Y)
        objective += opt_value
    objective /= K
    return objective

# # example usage:
# x = np.array([[1, 2, 3], [4, 5, 6]])  # Example m-dimensional vector
# input_measures = generate_gaussians(2, 3, seed=38)
# print(input_measures)
# objective = objective_compute(x, input_measures)
# print(objective)

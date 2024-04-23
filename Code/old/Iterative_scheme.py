import numpy as np
from Gaussian_generate import *
from KS_SCLS import *
from Objective import *
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import psutil
import math
import time

# Function to get CPU occupation
def get_cpu_occupation():
    return psutil.cpu_percent(interval=1)


# Specify the directory name
directory = "parameters"

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

###########################################################

# Generate a sample from the measure mu0
# INPUT:
# rho0 - tuple containing the mean vector and covariance matrix of the measure mu0;
# R - radius of the ball;
# OUTPUT:
# sample - sample from the measure mu0.
def rho0_sample_generate(rho0):
    sample = generate_gaussian_vectors(rho0, 1, seed=None)
    return sample.reshape(-1, 1)

def mu0_sample_generate(rho0, R):
    accept = 0
    while accept == 0:
        sample = generate_gaussian_vectors(rho0, 1, seed=None)
        if np.linalg.norm(sample) < R:
            accept = 1
    return sample.reshape(-1, 1) # d x 1

###########################################################
# Generate a sample from the measure in the iter-th iteration
# INPUT:
# rho0 - tuple containing the mean vector and covariance matrix of the measure mu0;
# R - radius of the ball;
# iter - number of iterations;
# OUTPUT:
# x - sample from the measure in the iter-th iteration.

def iterative_sample_generate(K, rho0_sample, iter):
    x = rho0_sample
    # breakpoint()
    for t in range(iter):
        hat_T_beta_sum = np.zeros(len(x)).reshape(-1, 1)
        for k in range(K):
            file_path = "parameters/output_{}_{}.json".format(t, k)
            BX, lambda_lower, lambda_upper, tilde_BG, V = read_output(file_path)
            # breakpoint()
            _, hat_T_beta = KS_SCLS_compute(x, BX, lambda_lower, 
            lambda_upper, 
            tilde_BG, 
            V, 
            Tau = 1000, 
            beta = 0.1)
            # breakpoint()
            hat_T_beta_sum += hat_T_beta
            # breakpoint()
        hat_T_beta_average = hat_T_beta_sum / K
        x = hat_T_beta_average
        # breakpoint()
    return x



###########################################################
# Generate samples from the measure mu_iter using the rejection sampling algorithm
# INPUT:
# rho0 - tuple containing the mean vector and covariance matrix of the measure mu0;
# R - radius of the ball;
# n_samples - number of samples to generate;
# OUTPUT:
# accepted - list of the accepted samples from the measure mu0.

def iterative_rejection_sampling(K, rho0, iter, n_samples, R = 1000):
    accepted = []
    while len(accepted) < n_samples:
        rho0_sample = rho0_sample_generate(rho0)
        sample = iterative_sample_generate(K, rho0_sample, iter)
        # breakpoint()
        if np.linalg.norm(sample) < R:
            accepted.append(sample)
        print("{}-th sample".format(len(accepted)))
        
    return np.array(accepted)[:, :, 0] #


###########################################################

def iterative_scheme(input_measures, rho0, n_samples):
    K = len(input_measures)
    old_objective = math.inf
    difference = math.inf
    objective_list = []
    iter = 0 # modify every time 
    while difference > 1e-3:
    #while abs(objective - old_objective) > 1e-3:
        # old_objective = objective.copy()
        # breakpoint()
        for k in range(len(input_measures)):
            # breakpoint()
            BX = iterative_rejection_sampling(K, rho0, iter, n_samples) # samples from mu_{iter}
            BY = generate_gaussian_vectors(input_measures[k], n_samples, seed=None) # samples from nu_k
            # breakpoint()

            lambda_lower = 0.1
            lambda_upper = 1000

            tilde_BG, V = KS_SCLS_construct(BX, BY, lambda_lower, lambda_upper)
            output_file = os.path.join("parameters", "output_{}_{}.json".format(iter, k))
            data = {
                "BX": BX.tolist(),
                "BY": BY.tolist(),
                "lambda_lower": lambda_lower,
                "lambda_upper": lambda_upper,
                "tilde_BG": tilde_BG.tolist(),
                "V": V.tolist()
            }
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=4)  # Serialize data to JSON with indentation for readability
            
            # breakpoint()

        output_sample = iterative_rejection_sampling(K, rho0, iter, n_samples)
        new_objective = objective_compute(output_sample, input_measures, n_samples)
        difference = abs(new_objective - old_objective)
        objective_list.append(new_objective)
        objective_file = os.path.join("parameters", "objective_list.json")
        obj_data = {"Objectives: ".format(iter): objective_list}
        with open(objective_file, 'w') as f:
            json.dump(obj_data, f)

        old_objective = new_objective
        iter += 1
        cpu_usage = get_cpu_occupation()
        print("Current CPU usage:", cpu_usage, "%")
        # breakpoint()

    return output_sample, objective_list

###########################################################
K = 5
dim = 2
input_measures = generate_gaussians(K, dim, seed=10)
rho0 = generate_gaussians(1, dim, seed=40)[0] 

output_sample, objective_list = iterative_scheme(input_measures, rho0, n_samples = 50)
print(output_sample)

# Create a vector of indices (x-axis)
x = range(1, len(output_sample) + 1)

# Plot the data
plt.plot(x, output_sample)

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('2D Plot of Output Sample')

# Show the plot
plt.show()

# breakpoint()

# def empirical_barycenter(input_measures, rho0, iter = 5, n_samples = 1000, R = 100):
#     K = len(input_measures)
#     empirical_barycenter = iterative_rejection_sampling(K, rho0, iter, n_samples, R)
#     return empirical_barycenter

# barycenter = empirical_barycenter(input_measures, rho0, iter = 5, n_samples = 1000)
# # print(barycenter)

# # Assuming barycenter is a numpy array with shape (n_samples, 3)
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# # Scatter plot
# ax.scatter(barycenter[:, 0], barycenter[:, 1], barycenter[:, 2], c='b', marker='o')

# Set labels and title
# ax.set_xlabel('Dimension 1')
# ax.set_ylabel('Dimension 2')
# ax.set_zlabel('Dimension 3')
# ax.set_title('3D Scatter Plot')

# plt.show()



############################################ TESTS ############################################
# sample = mu0_sample_generate(rho0, 100)
# print(sample)
# breakpoint()
# sample2 = iterative_sample_generate(5, sample, 0)
# print(sample2)
# breakpoint()
# accepted = iterative_rejection_sampling(0, rho0, 0, 3)
# print(accepted)
# breakpoint()
# accepted = iterative_rejection_sampling(5, rho0, 1, 4, R = 100)
# print(accepted)
# mu0_sample = mu0_sample_generate(rho0, 100)
# x = iterative_sample_generate(5, mu0_sample, 1)
# print(x)









    
   


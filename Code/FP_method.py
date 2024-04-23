import scipy as sp
from scipy.linalg import sqrtm, pinv, norm
import numpy as np
import sys, os

class FP_method:
    def __init__(self, dim, mu_list, Cov_list):
        self.dim = dim
        self.mu = mu_list
        self.Cov_list = Cov_list  

    def trace_list(self):
        trace_list = []
        for i in range(len(self.Cov_list)):
            trace_list.append(np.trace(self.Cov_list[i]))
        return trace_list
        
    def compute_bary_mean(self):
        mean = np.mean(self.mu, axis = 0)
        mean_cost = np.linalg.norm(self.mu - mean)**2 / len(self.mu)
        # breakpoint()
        return mean, mean_cost
    
    def compute_bary_cov(self, Sigma):
        Sigma_sum = np.zeros((self.dim, self.dim))
        for i in range(len(self.Cov_list)):
            sub_Sigma_square = sqrtm(Sigma) @ self.Cov_list[i] @ sqrtm(Sigma)
            sub_Sigma = sqrtm(sub_Sigma_square)
            Sigma_sum += sub_Sigma
        Sigma_sum = Sigma_sum / len(self.Cov_list)
        Sigma_update = np.linalg.solve(sqrtm(Sigma), np.eye(self.dim)) @ Sigma_sum @ Sigma_sum @ np.linalg.solve(sqrtm(Sigma), np.eye(self.dim))
        return Sigma_update
    
    def compute_V(self, Sigma):
        trace2_list = []
        for i in range(len(self.Cov_list)):
            sub_Sigma_square = sqrtm(Sigma) @ self.Cov_list[i] @ sqrtm(Sigma)
            trace2_list.append(np.trace(sqrtm(sub_Sigma_square)))
        V = np.trace(Sigma) + np.mean(self.trace_list()) - 2 * np.mean(trace2_list)
        return V

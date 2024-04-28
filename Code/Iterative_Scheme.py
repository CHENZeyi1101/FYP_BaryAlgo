import numpy as np
import gurobipy as gp
from gurobipy import GRB
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import psutil
import math
import time
import scipy as sp
from scipy.linalg import sqrtm, pinv, norm

def save_data(data, pathname = None, filename = None):
    output_file = os.path.join(pathname, filename)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4) 

def read_data(pathname = None, filename = None):
    file_path = os.path.join(pathname, filename)
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

class Measure:
    def __init__(self, dim, distribution_type = None, parameters = None):
        self.dim = dim
        self.type = distribution_type
        self.parameters = parameters

    def generate_random_parameters(self, num_measures, seed = 41):
        dim = self.dim
        np.random.seed(seed)
        parameters = []
        for _ in range(num_measures):
            mean = np.random.rand(dim) * 10
            A = np.random.rand(dim, dim)
            A = np.dot(A, A.T) * 10 + np.eye(dim) # Ensure covariance matrix is positive definite
            parameters.append((mean, A))
        return parameters

    def generate_gaussian_sample(self, size, k, seed = None):
        np.random.seed(seed)
        if self.type == "gaussian":
            mean, covariance_matrix = self.parameters[k]
            return np.random.multivariate_normal(mean, covariance_matrix, size)
        else:
            raise ValueError("Unsupported distribution type")
        
    def generate_mixture_gaussian_sample(self, size, seed = None):
        np.random.seed(seed)
        if self.type == "mixture_gaussian":
            gaussians = self.parameters
            n_mix = len(gaussians)
            U = np.random.rand(size)
            samples = []
            for u in U:
                group = math.floor(u * n_mix)
                mean, covariance_matrix = gaussians[group]
                sample = np.random.multivariate_normal(mean, covariance_matrix)
                samples.append(sample)
            return np.array(samples)
        else:
            raise ValueError("Unsupported distribution type")
        
    def generate_truncated_sample(self, size, R = 1, A = None, b = None, seed = None):
        dim = self.dim
        if self.type == "gaussian": # location scattered
            accepted = []
            sample = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), 100 * size)
            index = 0
            while len(accepted) < size:
                if norm(sample[index]) < R: # Truncatiion
                    accepted.append(A @ sample[index] + b)
                index += 1
            return np.squeeze(np.array(accepted))

        elif self.type == "mixture_gaussian": # not location scattered
            accepted = []
            while len(accepted) < size:
                sample = self.generate_mixture_gaussian_sample(100 * size, seed)
                index = 0
                while len(accepted) < size and index < 10 * size:
                    if norm(sample[index]) < R:
                        accepted.append(sample[index])
                    index += 1
            return np.squeeze(np.array(accepted))
        
        else:
            raise ValueError("Unsupported distribution type")

class OT_Map_Estimator:
    def __init__(self, BX, BY, lambda_lower = 0.1, lambda_upper = 1000):
        self.BX = BX # Samples from the source measure
        self.BY = BY # Samples from the target measure
        self.lambda_lower = lambda_lower
        self.lambda_upper = lambda_upper

    def solve_OptCoupling_matrix(self):
        x, y = self.BX, self.BY
        m, n = len(self.BX), len(self.BY)
        model = gp.Model("LP_OptCoupling")
        
        # Define the decision variables
        pi = {}
        for i in range(m):
            for j in range(n):
                pi[i, j] = model.addVar(lb=0.0, ub = 1.0, vtype=GRB.CONTINUOUS, name=f"pi_{i}_{j}")
        model.update()
        
        # Define the objective function
        obj = gp.quicksum(pi[i, j] * np.linalg.norm(x[i] - y[j])**2 for i in range(m) for j in range(n))
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Add constraints: sum(pi_ij) = 1/n for all j
        for j in range(n):
            model.addConstr(gp.quicksum(pi[i, j] for i in range(m)) == 1/n)
        # Add constraints: sum(pi_ij) = 1/m for all i
        for i in range(m):
            model.addConstr(gp.quicksum(pi[i, j] for j in range(n)) == 1/m)
        
        # Optimize the model
        model.optimize()

        optimal_solution = np.array([[pi[i, j].x for j in range(n)] for i in range(m)])
        optimal_objective = model.objVal
        
        return optimal_solution, optimal_objective

    def solve_opt_tuples_location_scatter(self, A, b, R):
        BX, BY = self.BX, self.BY
        lambda_lower, lambda_upper = self.lambda_lower, self.lambda_upper
        hat_pi_star, objective = self.solve_OptCoupling_matrix()
        m, n = len(BX), len(BY)
        dim = len(BX[0])
        
        model = gp.Model("OptTuple_qcqp")
        
        # Set NumericFocus parameter to 2 (Aggressive numerical emphasis)
        # model.setParam('NumericFocus', 2)
        
        tilde_BIg = {}
        tilde_varphi = {}
        for i in range(m):
            tilde_BIg[i] = model.addMVar(shape = (dim,), lb=-GRB.INFINITY, name="tilde_BIg_{}".format(i))
            tilde_varphi[i] = model.addVar(lb=-GRB.INFINITY, name="tilde_varphi_{}".format(i))
        model.update()
        
        # Define the objective function
        obj_expr = gp.QuadExpr()
        for i in range(m):
            for j in range(n):
                if hat_pi_star[i][j] > 1e-8:
                    obj_expr += ((BY[j] - tilde_BIg[i] - lambda_lower * BX[i]) @ (BY[j] - tilde_BIg[i]- lambda_lower * BX[i])) * hat_pi_star[i][j]
                else:
                    pass

        model.setObjective(obj_expr, GRB.MINIMIZE)
        
        # Add constraints
        for i in range(m):
            aux = tilde_BIg[i] + lambda_lower * BX[i]
            # model.addConstr(aux @ aux <= R^2)
            A_inv = sp.linalg.solve(A, np.eye(dim))
            model.addConstr((A_inv @ (aux - b)) @ (A_inv @ (aux - b)) <= R^2)
            # breakpoint()
            for j in range(m):
                if i != j:
                    constraint_expr = gp.QuadExpr()
                    inner_product = tilde_BIg[i]@(BX[j] - BX[i])
                    norm_squared_tilde_BIg = (tilde_BIg[i] - tilde_BIg[j])@(tilde_BIg[i] - tilde_BIg[j])
                    # breakpoint()
                    constraint_expr += tilde_varphi[i] - tilde_varphi[j] + inner_product + norm_squared_tilde_BIg / (2*(lambda_upper - lambda_lower))
                    model.addConstr(constraint_expr <= 0, "constraint_{}_{}".format(i, j))
                else:
                    pass
                    
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            print("Optimal solution found")
            optimal_tilde_BIg = np.array([[tilde_BIg[i][j].x for j in range(len(BY[0]))] for i in range(m)])
            optimal_tilde_varphi = np.array([tilde_varphi[i].x for i in range(m)])
        else:
            print("No optimal solution found")

        return objective, optimal_tilde_BIg, optimal_tilde_varphi # optimal_BIG.shape = (m x d)
    
    def solve_opt_tuples_general(self, R = 1):
        BX, BY = self.BX, self.BY
        lambda_lower, lambda_upper = self.lambda_lower, self.lambda_upper
        hat_pi_star, objective = self.solve_OptCoupling_matrix()
        m, n = len(BX), len(BY)
        dim = len(BX[0])
        
        model = gp.Model("OptTuple_qcqp")
        
        # Set NumericFocus parameter to 2 (Aggressive numerical emphasis)
        # model.setParam('NumericFocus', 2)
        
        tilde_BIg = {}
        tilde_varphi = {}
        for i in range(m):
            tilde_BIg[i] = model.addMVar(shape = (dim,), lb=-GRB.INFINITY, name="tilde_BIg_{}".format(i))
            tilde_varphi[i] = model.addVar(lb=-GRB.INFINITY, name="tilde_varphi_{}".format(i))
        model.update()
        
        # Define the objective function
        obj_expr = gp.QuadExpr()
        for i in range(m):
            for j in range(n):
                if hat_pi_star[i][j] > 1e-8:
                    obj_expr += ((BY[j] - tilde_BIg[i] - lambda_lower * BX[i]) @ (BY[j] - tilde_BIg[i]- lambda_lower * BX[i])) * hat_pi_star[i][j]
                else:
                    pass

        model.setObjective(obj_expr, GRB.MINIMIZE)
        
        # Add constraints
        for i in range(m):
            aux = tilde_BIg[i] + lambda_lower * BX[i]
            model.addConstr(aux @ aux <= R)
            # breakpoint()
            for j in range(m):
                if i != j:
                    constraint_expr = gp.QuadExpr()
                    inner_product = tilde_BIg[i]@(BX[j] - BX[i])
                    norm_squared_tilde_BIg = (tilde_BIg[i] - tilde_BIg[j])@(tilde_BIg[i] - tilde_BIg[j])
                    # breakpoint()
                    constraint_expr += tilde_varphi[i] - tilde_varphi[j] + inner_product + norm_squared_tilde_BIg / (2*(lambda_upper - lambda_lower))
                    model.addConstr(constraint_expr <= 0, "constraint_{}_{}".format(i, j))
                else:
                    pass
                    
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            print("Optimal solution found")
            optimal_tilde_BIg = np.array([[tilde_BIg[i][j].x for j in range(len(BY[0]))] for i in range(m)])
            optimal_tilde_varphi = np.array([tilde_varphi[i].x for i in range(m)])
        else:
            print("No optimal solution found")
        return objective, optimal_tilde_BIg, optimal_tilde_varphi # optimal_BIG.shape = (m x d)
    
    def interp_QP(self, input_vector, optimal_tilde_BIg, optimal_tilde_varphi):
        lambda_lower, lambda_upper = self.lambda_lower, self.lambda_upper
        BX = self.BX
        m = len(BX)
        x = input_vector
        optimal_BIg = optimal_tilde_BIg + lambda_lower * BX
        tilde_BG = optimal_tilde_BIg.T
        BIg_square = np.diag(np.dot(optimal_BIg, optimal_BIg.T))
        BX_square = np.diag(np.dot(BX, BX.T))
        BIg_BX = np.diag(np.dot(optimal_BIg, BX.T))
        v_star = optimal_tilde_varphi + BIg_square / (2 *(lambda_upper - lambda_lower)) + BX_square * lambda_lower * lambda_upper / (2 *(lambda_upper - lambda_lower)) - lambda_upper * BIg_BX / (lambda_upper - lambda_lower)

        model = gp.Model("QP")

        # Define the variables
        BIw = {}
        BIw = model.addMVar(shape = (m,), lb = 0, ub = 1, name = "BIw") # BIw as \R^m-vector

        # Define the objective function
        obj_expr = gp.QuadExpr()
        tilde_BG_x_V = tilde_BG.T @ x + v_star 
        innerprod = tilde_BG_x_V.T @ BIw
        norm_Gw = (tilde_BG @ BIw) @ (tilde_BG @ BIw)
        obj_expr += innerprod - (1 / (2 * (lambda_upper - lambda_lower))) * norm_Gw
        model.setObjective(obj_expr, GRB.MAXIMIZE)

        # Define the constraints
        for i in range(m):
            model.addConstr(BIw.sum() == 1)

        model.optimize()

        if model.status == GRB.OPTIMAL:
            optimal_weight = np.array(BIw.X)
            optimal_objective = model.ObjVal
        else:
            print("No optimal solution found")

        tilde_varphi_star = optimal_objective
        tilde_BIg_star = np.dot(tilde_BG, optimal_weight)

        return tilde_BIg_star, tilde_varphi_star
    
    
    def KS_SCLS(self, input_vector, beta, optimal_tilde_BIg, optimal_tilde_varphi, Tau = 1000):
        lambda_lower, _ = self.lambda_lower, self.lambda_upper
        x = input_vector
        X = np.tile(x, (Tau, 1))
        BX = self.BX
        dim = len(BX[0])
        eta = np.random.multivariate_normal(np.zeros(dim), beta * np.eye(dim), Tau)
        X_eta = X + eta
        value = 0
        map_list = np.zeros(dim)

        for t in range(Tau):
            tilde_BIg_star, tilde_varphi_star = self.interp_QP(X_eta[t], optimal_tilde_BIg, optimal_tilde_varphi)
            value += tilde_varphi_star
            map_list += tilde_BIg_star
        
        tilde_varphi_beta = value / Tau
        tilde_BIg_beta = map_list / Tau

        varphi_beta = tilde_varphi_beta + (lambda_lower / 2) * np.linalg.norm(x)**2
        T_beta = tilde_BIg_beta + lambda_lower * x

        return varphi_beta, T_beta

class Iterative_Scheme:
    def __init__(self, dim, K, n_samples, R = 100, lambda_lower = 0.1, lambda_upper = 1000):
        self.dim = dim
        # self.rho0 = Measure("gaussian", [(np.zeros(dim), np.eye(dim))])
        # self.OT_map_estimator = OT_Map_Estimator
        self.K = K
        self.n_samples = n_samples
        self.R = R
        self.lambda_lower = lambda_lower
        self.lambda_upper = lambda_upper

    def save_data(self, data, pathname = None, filename = None):
        output_file = os.path.join(pathname, filename)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4) 

    def read_data(self, pathname = None, filename = None):
        file_path = os.path.join(pathname, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    
    def rejection_sampling(self, iter, pathname = None):
        count = 0
        accepted = []
        while count < self.n_samples:
            sample = np.random.multivariate_normal(np.zeros(self.dim), np.eye(self.dim))
            for t in range(iter): # pass (i.e., no mapping) when t = 0
                sum_sample = np.zeros(self.dim)
                for k in range(self.K):
                    data = self.read_data(pathname, "output_{}_{}.json".format(t, k))
                    BX, BY, _, optimal_tilde_BIg, optimal_tilde_varphi = np.asarray(data["BX"]), np.asarray(data["BY"]), data["objective"], np.asarray(data["optimal_tilde_BIg"]), np.asarray(data["optimal_tilde_varphi"])
                    OT_map_estimator = OT_Map_Estimator(BX, BY, self.lambda_lower, self.lambda_upper)
                    _, branch_sample = OT_map_estimator.KS_SCLS(
                        sample, 0.01, optimal_tilde_BIg, optimal_tilde_varphi, Tau = 200)
                    sum_sample += branch_sample
                sample = sum_sample / self.K
            if np.linalg.norm(sample) < self.R:
                count += 1
                accepted.append(sample.tolist())
        return np.array(accepted)
    
    def W2_square(self, BX, BY):
        W2_square = OT_Map_Estimator(BX, BY).solve_OptCoupling_matrix()[1]
        return W2_square
    
    def map_construct_location_scatter(self, BX, BY, A, b, R, iter, k, pathname = None):
        OT_map_estimator = OT_Map_Estimator(BX, BY, self.lambda_lower, self.lambda_upper)
        objective, optimal_tilde_BIg, optimal_tilde_varphi = OT_map_estimator.solve_opt_tuples_location_scatter(A, b, R)
        data = {
            "BX": BX.tolist(),
            "BY": BY.tolist(),
            "objective": objective,
            "optimal_tilde_BIg": optimal_tilde_BIg.tolist(),
            "optimal_tilde_varphi": optimal_tilde_varphi.tolist()
        }
        self.save_data(data, pathname, "output_{}_{}.json".format(iter, k))

    def map_construct_general(self, BX, BY, R, iter, k, pathname = None):
        OT_map_estimator = OT_Map_Estimator(BX, BY, self.lambda_lower, self.lambda_upper)
        objective, optimal_tilde_BIg, optimal_tilde_varphi = OT_map_estimator.solve_opt_tuples_general(R)
        data = {
            "BX": BX.tolist(),
            "BY": BY.tolist(),
            "objective": objective,
            "optimal_tilde_BIg": optimal_tilde_BIg.tolist(),
            "optimal_tilde_varphi": optimal_tilde_varphi.tolist()
        }
        self.save_data(data, pathname, "output_{}_{}.json".format(iter, k))

    def present_barycenter(self, iter, pathname = None):
        barycenter = self.rejection_sampling(iter, pathname)
        return barycenter
        

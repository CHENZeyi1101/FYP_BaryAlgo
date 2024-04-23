import gurobipy as gp
from gurobipy import GRB
from scipy import sparse
import numpy as np
from SC_OptTuple import *
from LP_OptCoupling import *

def MVar_to_array(MVar):
    MVar_list = list(MVar.items())
    MVar_array = np.transpose(np.array([el[-1] for el in MVar_list]))
    MVar_array = sparse.csr_matrix(MVar_array)
    return MVar_array

####### Define the notations as in the paper #######
# INPUT:
# Opt_BIg - optimal solution for the BIg variables
# Opt_varphi - optimal solution for the varphi variables
# BX - m x d matrix (numpy array) containing the vectors from the first set
# lambda_lower - lower bound for the lambda parameter
# lambda_upper - upper bound for the lambda parameter

# OUTPUT:
# tilde_BG - m x d matrix (numpy array) containing the optimal gradient values
# V - m-dimensional vector (numpy array) containing the optimal function values

def notation_star(Opt_BIg, Opt_varphi, BX, lambda_lower, lambda_upper):
    m = len(BX)
    tilde_BG = Opt_BIg.copy().T
    V = np.zeros(m).reshape(-1, 1)

    for i in range(m):
        tilde_BG[:, i] = (Opt_BIg[i] - lambda_lower * BX[i])
        # breakpoint()
        v_i = Opt_varphi[i] + 1/(2*(lambda_upper - lambda_lower)) * np.linalg.norm(Opt_BIg[i])**2 + (lambda_lower * lambda_upper / (2 * (lambda_upper - lambda_lower))) * np.linalg.norm(BX[i])**2 - float(lambda_lower / (lambda_upper - lambda_lower)) * (Opt_BIg[i]@ BX[i])
        V[i] = v_i
        # breakpoint()

    return tilde_BG, V

####### Interpolate the optimal solution using the QP model #######

# INPUT:
# x - d-dimensional vector (numpy array)
# tilde_BG - m x d matrix (numpy array) containing the optimal gradient values
# tilde_varphi - m-dimensional vector (numpy array) containing the optimal function values
# BX - m x d matrix (numpy array) containing the vectors from the first set
# lambda_lower - lower bound for the lambda parameter
# lambda_upper - upper bound for the lambda parameter

# OUTPUT:
# tilde_BIg_star - interpolated optimal gradient for the SC-LS
# tilde_varphi_star - interpolated optimal function for the SC-LS

def Interp_QP(x, BX, tilde_BG, V, lambda_lower, lambda_upper):
    m = len(BX)
    
    # Define the model
    model = gp.Model("QP")

    # Define the variables
    BIw = {}
    BIw = model.addMVar((m,), lb = 0, ub = 1, name = "BIw") # BIw as \R^m-vector

    # Define the objective function
    obj_expr = gp.QuadExpr()
    # if tilde_BG.T.shape[1] != 2:
    #     breakpoint()
    tilde_BG_x_V = tilde_BG.T @ x + V 
    innerprod = tilde_BG_x_V.T @ BIw
    norm_Gw = (tilde_BG @ BIw) @ (tilde_BG @ BIw)
    obj_expr += innerprod - (1 / (2 * (lambda_upper - lambda_lower))) * norm_Gw
    model.setObjective(obj_expr, GRB.MAXIMIZE)

    # Define the constraints
    for i in range(m):
        model.addConstr(BIw.sum() == 1)

    # Optimize the model
    model.optimize()

    # Retrieve and print the optimal solution
    if model.status == GRB.OPTIMAL:
        # print(type(BIw))
        optimal_weight = BIw.X
        optimal_objective = model.ObjVal  # Optimal objective value
        # print("Optimal weight:", optimal_weight)
        # print("Optimal objective:", optimal_objective)
    else:
        print("No optimal solution found")

    tilde_varphi_star = optimal_objective
    tilde_BIg_star = tilde_BG @ optimal_weight
    print("tilde_BIg_star", tilde_BIg_star)
    print("tilde_varphi_star", tilde_varphi_star)

    return tilde_BIg_star, tilde_varphi_star


# BY = np.random.randint(100, size = (20, 2))  # Example BY matrix
# BX = np.random.randint(100, size = (20, 2)) # Example BX matrix
# hat_pi_star = solve_OptCoupling_matrix(BX, BY)[0]  # Example hat_pi_star matrix
# lambda_lower = 0.1 # Example lower bound for lambda
# lambda_upper = 1000 # Example upper bound for lambda

# breakpoint()

# opt_BIg, opt_varphi= solve_qcqp(BX, BY, hat_pi_star, lambda_lower, lambda_upper)

# breakpoint()

# tilde_BG, V = notation_star(opt_BIg, opt_varphi, BX, lambda_lower, lambda_upper)
# print(tilde_BG)
# print(V)

# x = np.array([1, 2, 3, 4]).reshape(-1, 1)
# tilde_BIg_star, tilde_varphi_star = Interp_QP(x, BX, tilde_BG, V, lambda_lower, lambda_upper)
# breakpoint()



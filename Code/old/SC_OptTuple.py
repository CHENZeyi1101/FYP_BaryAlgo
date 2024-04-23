import gurobipy as gp
from gurobipy import GRB
import numpy as np


# Solve the QCQP to compute the optimal tuple between two sets of vectors BX and BY (according to the OptTuple algorithm)
# using the given optimal coupling matrix \hat{\pi}^\star and the given lambda_lower and lambda_upper values
# INPUT:
# BX - m x d matrix (numpy array) containing the vectors from the first set
# BY - n x d matrix (numpy array) containing the vectors from the second set
# hat_pi_star - m x n matrix (numpy array) containing the optimal coupling matrix
# lambda_lower - lower bound for the lambda parameter
# lambda_upper - upper bound for the lambda parameter
# OUTPUT:
# optimal_solution_BIg - optimal solution for the BIg variables
# optimal_solution_varphi - optimal solution for the varphi variables
# objective_values - list of objective values at each iteration


def solve_qcqp(BX, BY, hat_pi_star, lambda_lower, lambda_upper):

    # breakpoint()
    m, n = len(BX), len(BY)
    # d = len(BX[0])
    # print(hat_pi_star[1][1])
    
    # Create a new model
    model = gp.Model("OptTuple_qcqp")
    
    # Set NumericFocus parameter to 3 (Aggressive numerical emphasis)
    model.setParam('NumericFocus', 2)
    
    # Create decision variables
    tilde_BIg = {}
    tilde_varphi = {}
    for i in range(m):
        tilde_BIg[i] = model.addMVar((len(BY[0]),), lb=-GRB.INFINITY)
        tilde_varphi[i] = model.addVar(lb=-GRB.INFINITY)
        # \varphi_i as scalar.
    
    # Update model to integrate new variables
    model.update()
    
    # Set objective function
    obj_expr = gp.QuadExpr()
    for i in range(m):
        for j in range(n):
            if hat_pi_star[i][j] > 1e-5:
                obj_expr += ((BY[j] - tilde_BIg[i] - lambda_lower * BX[i])@(BY[j] - tilde_BIg[i]- lambda_lower * BX[i])) * hat_pi_star[i][j]
            else:
                pass

    model.setObjective(obj_expr, GRB.MINIMIZE)
    # breakpoint()
    
    # Add constraints
    for i in range(m):
        for j in range(m):
            if i != j:
                constraint_expr = gp.QuadExpr()
                inner_product = tilde_BIg[i]@(BX[j] - BX[i])
                norm_squared_tilde_BIg = (tilde_BIg[i] - tilde_BIg[j])@(tilde_BIg[i] - tilde_BIg[j])
                constraint_expr += tilde_varphi[i] -tilde_varphi[j] + inner_product +   (1 / (2*(lambda_upper - lambda_lower))) * norm_squared_tilde_BIg
                model.addConstr(constraint_expr <= 0, "constraint_{}_{}".format(i, j))
                
    # Optimize the model
    model.optimize()
    # breakpoint()
    
    # Retrieve and print the optimal solution
    if model.status == GRB.OPTIMAL:
        optimal_BIg = np.array([[tilde_BIg[i][j].x + lambda_lower * BX[i][j] for j in range(len(BY[0]))] for i in range(m)])
        optimal_varphi = np.array([tilde_varphi[i].x + lambda_lower**2 / 2 * np.linalg.norm(BX[i])**2 for i in range(m)])

    else:
        print("No optimal solution found")

    return optimal_BIg, optimal_varphi # optimal_BIG.shape = (m x d)

# Example usage:
# BY = np.random.randint(100, size = (30, 2))  # Example BY matrix
# BX = np.random.randint(100, size = (30, 2)) # Example BX matrix
# # BY = np.array([[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9], [5, 4, 3, 2, 4, 1], [2, 3, 4, 4, 3, 1], [24, 8, 9, 4, 3, 21]])  # Example BY matrix
# # BX = np.array([[0, 0, 0, 2, 3, 5], [1, 1, 1, 6, 3, 9], [4, 4, 4, 4, 4, 5]])  # Example BX matrix
# # breakpoint()
# hat_pi_star = np.eye(30)
# # hat_pi_star = np.full((len(BX), len(BY)), 1/(len(BX)*len(BY)))  # Example hat_pi_star matrix
# lambda_lower = 0.1  # Example lower bound for lambda
# lambda_upper = 1000  # Example upper bound for lambda

# opt_BIg, opt_varphi = solve_qcqp(BX, BY, hat_pi_star, lambda_lower, lambda_upper)
# print(opt_BIg.shape)

import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Solve the linear program to compute the optimal coupling between two sets of vectors x and y
# using the Euclidean distance as the cost function
# INPUT:
# x - m-dimensional vector (numpy array) (m x d)
# y - n-dimensional vector (numpy array) (n x d)
# OUTPUT:
# optimal_solution - optimal coupling matrix (numpy array) (\widehat{\pi}^\star)


def solve_OptCoupling_matrix(x, y):
    m = len(x)
    n = len(y)
    
    # Create a new model
    model = gp.Model("LP_OptCoupling")
    
    # Create decision variables
    pi = {}
    for i in range(m):
        for j in range(n):
            pi[i, j] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"pi_{i}_{j}")
    
    # Update model to integrate new variables
    model.update()
    
    # Set objective function: minimize sum(pi_ij * ||x_i - y_j||^2)
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
    # breakpoint()
    optimal_solution = np.array([[pi[i, j].x for j in range(n)] for i in range(m)])

    # Output the optimal objective
    optimal_objective = model.objVal
    
    
    return optimal_solution, optimal_objective

# # Example usage:
# x = np.array([[1, 2, 3], [4, 5, 6]])  # Example m-dimensional vector
# y = np.array([[0, 0, 0], [1, 1, 1]])  # Example n-dimensional vector
# solution, objective = solve_OptCoupling_matrix(x, y)

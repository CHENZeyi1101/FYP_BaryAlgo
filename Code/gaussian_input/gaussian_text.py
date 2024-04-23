import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from Iterative_Scheme import * 

# for seed in range(100):
#     print("seed = ", seed + 30)
#     dim = 2
#     size = 5
#     nu = Measure("gaussian", [(np.array([0, 0]), np.eye(dim))])
#     paras = nu.generate_random_parameters(1, dim, seed = seed + 20)
#     A = paras[0][1] /1000
#     # print(A)
#     b = paras[0][0]
#     if np.linalg.det(A) < 1:
#         matrix_save = {}
#         matrix_save['A_{}'.format(seed)] = A.tolist()
#         matrix_save['b_{}'.format(seed)] = b.tolist()
#         save_data(data = matrix_save, pathname = "gaussian_input", filename = "matrix_save2.json")
#         # breakpoint()
#         # A = np.eye(dim)
#         # b = np.zeros(dim)
#         dim = 2
#         K = 4
#         n_samples = 10
#         R = 50,
#         lambda_lower = 0.01
#         lambda_upper = 1000
#         iter_scheme = Iterative_Scheme(dim, K, n_samples, R, lambda_lower, lambda_upper)
#         BX = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), size)
#         BX = iter_scheme.rejection_sampling(0)
#         # breakpoint()
#         # normal = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), size)
#         # breakpoint()
#         # BY = (A @ normal.T).T + b
#         BY = nu.generate_truncated_sample(size, 1, A, b)
#         # print((pinv(A) @ (BY - b).T).T)
#         # breakpoint()
#         OT_map = OT_Map_Estimator(BX, BY, A, b)
#         OT_map.solve_opt_tuples()
#     else:
#         pass

dim = 2
size = 5
nu = Measure("gaussian", [(np.array([0, 0]), np.eye(dim))])
A, b = read_data(pathname = "gaussian_input", filename = "matrix_save1.json").values()
A = np.array(A)
b = np.array(b)
breakpoint()
# A = np.eye(dim)
# b = np.zeros(dim)
dim = 2
K = 4
n_samples = 10
R = 50,
lambda_lower = 0.01
lambda_upper = 1000
iter_scheme = Iterative_Scheme(dim, K, n_samples, R, lambda_lower, lambda_upper)
BX = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), size)
BX = iter_scheme.rejection_sampling(0)
# breakpoint()
# normal = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), size)
# breakpoint()
# BY = (A @ normal.T).T + b
BY = nu.generate_truncated_sample(size, 1, A, b)
# print((pinv(A) @ (BY - b).T).T)
# breakpoint()
OT_map = OT_Map_Estimator(BX, BY, A, b)
OT_map.solve_opt_tuples()


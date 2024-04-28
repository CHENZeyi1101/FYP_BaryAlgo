import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from Iterative_Scheme import * 

dim = 2
K = 4
n_samples = 100
R = 400
lambda_lower = 0.001
lambda_upper = 1000
iter_scheme = Iterative_Scheme(dim, K, n_samples, R, lambda_lower, lambda_upper)
measure = Measure(dim, "gaussian", [])
base_mu = np.zeros(dim)
base_sample = measure.generate_truncated_sample(size = 2000, R = 1, A = np.eye(dim), b = np.zeros(dim))
base_cov = (base_sample - base_mu).T @ (base_sample - base_mu) / 1999
base_A = sqrtm(base_cov)

nu_para = measure.generate_random_parameters(K, seed = 41)
dict_info = {}
dict_info['Dimension'] = dim
dict_info['Number of measures'] = K
dict_info["Generating seed"] = 41
dict_info['Number of samples per iteration'] = n_samples
dict_info['R'] = R
dict_info['lambda_lower'] = lambda_lower
dict_info['lambda_upper'] = lambda_upper
for i in range(K):
    mean, A_matrix = nu_para[i]
    dict_info['Mean of nu_{}'.format(i)] = mean.tolist()
    dict_info['A_matrix of nu_{}'.format(i)] = A_matrix.tolist()

save_data(data = dict_info, pathname = "gaussian_input/iter_data", filename = "info.json")

nu_list = []
for i in range(K):
    nu_list.append(Measure(dim, "gaussian", [nu_para[i]]))

iter = 0
radius_truncate = 1
difference = math.inf
objective_list = [math.inf]
objective_dict = {}
error = 1e-3
threshold = 6
while difference > error and iter < threshold:
    BX= iter_scheme.rejection_sampling(iter, pathname="gaussian_input/iter_data")
    V_value = 0
    for k in range(K):
        b, A = nu_list[k].parameters[0][0], nu_list[k].parameters[0][1]
        BY = nu_list[k].generate_truncated_sample(size = n_samples, R = radius_truncate, A = A @ pinv(base_A), b = b)  
        W2_square = iter_scheme.W2_square(BX, BY)
        V_value += W2_square
        iter_scheme.map_construct_location_scatter(BX, BY, A @ pinv(base_A), b, radius_truncate, iter, k, pathname = "gaussian_input/iter_data")
    objective = V_value / K
    difference = abs(objective - objective_list[-1])
    objective_list.append(objective)
    objective_dict['objective in iteration_{}'.format(iter)] = objective
    # objective_dict['difference in iteration_{}'.format(iter)] = difference
    save_data(data = objective_dict, pathname = "gaussian_input/iter_data", filename = "iter_objective.json")
    iter += 1

barycenter = iter_scheme.present_barycenter(iter = iter, pathname="gaussian_input/iter_data")
save_data(data = barycenter.tolist(), pathname = "gaussian_input/iter_data", filename = "barycenter.json")
BX = read_data(pathname = "gaussian_input/iter_data", filename = "barycenter.json")
try_sample = 0
bary = []
while try_sample < 100:
    V_value = 0
    for k in range(K):
        b = nu_list[k].parameters[0][0]
        A = nu_list[k].parameters[0][1]
        BY = nu_list[k].generate_truncated_sample(size = 200, R = 1, A = A @ pinv(base_A), b = b)
        W2_square = iter_scheme.W2_square(BX, BY)
        V_value += W2_square
    V_value = V_value / K
    bary_objective = V_value
    bary.append(bary_objective)
    try_sample += 1
save_data(data = bary_objective, pathname = "gaussian_input/iter_data", filename = "barycenter_objective.json")


# # barycenter = iter_scheme.present_barycenter(iter = 3, pathname="gaussian_input/iter_data")
# # save_data(data = barycenter.tolist(), pathname = "gaussian_input/iter_data", filename = "barycenter2.json")
# # save_data(data = barycenter.tolist(), pathname = "gaussian_input/iter_data", filename = "barycenter2.json")
# BX = read_data(pathname = "gaussian_input/iter_data", filename = "barycenter.json")
# # BX = read_data(pathname = "gaussian_input/iter_data", filename = "barycenter2.json")
# try_sample = 0
# bary = []
# while try_sample < 100:
#     V_value = 0
#     for k in range(K):
#         b = nu_list[k].parameters[0][0]
#         A = nu_list[k].parameters[0][1]
#         BY = nu_list[k].generate_truncated_sample(size = 200, R = 1, A = A @ pinv(base_A), b = b)
#         W2_square = iter_scheme.W2_square(BX, BY)
#         V_value += W2_square
#     V_value = V_value / K
#     bary_objective = V_value
#     bary.append(bary_objective)
#     try_sample += 1
# save_data(data = bary_objective, pathname = "gaussian_input/iter_data", filename = "barycenter_objective.json")



                
                    
                    

                
                
                


    

        



import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from Iterative_Scheme import * 

dim = 2
K = 4
n_samples = 100
radius_truncate = 5
R = 50
lambda_lower = 0.01
lambda_upper = 1000
iter_scheme = Iterative_Scheme(dim, K, n_samples, R, lambda_lower, lambda_upper)
measure = Measure(dim)
n_mix = 5
dict_info = {}
dict_info['Dimension'] = dim
dict_info['Number of measures'] = K
dict_info['Number of mixture components'] = n_mix
dict_info['Radius of truncation'] = radius_truncate
dict_info["Generating seed"] = 100
dict_info['Number of samples per iteration'] = n_samples
dict_info['R'] = R
dict_info['lambda_lower'] = lambda_lower
dict_info['lambda_upper'] = lambda_upper

nu_list = []
for i in range(K):
    gaussian_mix = measure.generate_random_parameters(n_mix, seed = 100 + i)
    print(gaussian_mix)
    nu = Measure(dim, "mixture_gaussian", gaussian_mix)
    nu_list.append(nu)
    dict_info["mixture components of nu_{}".format(i)] = [(x.tolist(), y.tolist()) for x, y in gaussian_mix]

save_data(data = dict_info, pathname = "mixture_input/iter_data", filename = "info.json")

iter = 0
difference = math.inf
objective_list = [math.inf]
objective_dict = {}
error = 1e-3
threshold = 6

while difference > error and iter < threshold:
    BX= iter_scheme.rejection_sampling(iter, pathname="mixture_input/iter_data")
    V_value = 0
    for k in range(K):
        BY = nu_list[k].generate_truncated_sample(size = n_samples, R = radius_truncate)
        # breakpoint()   
        W2_square = iter_scheme.W2_square(BX, BY)
        V_value += W2_square
        iter_scheme.map_construct_general(BX, BY, radius_truncate, iter, k, pathname = "mixture_input/iter_data")

    objective = V_value / K
    difference = abs(objective - objective_list[-1])
    objective_list.append(objective)
    objective_dict['objective in iteration_{}'.format(iter)] = objective
    # objective_dict['difference in iteration_{}'.format(iter)] = difference
    save_data(data = objective_dict, pathname = "mixture_input/iter_data", filename = "iter_objective.json")
    iter += 1
#iter = 10
barycenter = iter_scheme.present_barycenter(iter = iter, pathname="mixture_input/iter_data")
save_data(data = barycenter.tolist(), pathname = "mixture_input/iter_data", filename = "barycenter.json")

BX = read_data(pathname = "mixture_input/iter_data", filename = "barycenter.json")
V_value = 0
for k in range(K):
    BY = nu_list[k].generate_truncated_sample(size = n_samples, R = radius_truncate)
    W2_square = iter_scheme.W2_square(BX, BY)
    V_value += W2_square
V_value = V_value / K
bary_objective = V_value
save_data(data = bary_objective, pathname = "mixture_input/iter_data", filename = "barycenter_objective.json")



                
                    
                    

                
                
                


    

        



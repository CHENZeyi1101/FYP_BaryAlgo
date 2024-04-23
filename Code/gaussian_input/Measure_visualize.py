import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from Iterative_Scheme import * 
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

def gaussian_density(x, mu, sigma):
        return 1 / (np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def plot_TN_density_surface(scaling_constant, A, b, xrange=(-10, 20), yrange=(-10, 20), grid_size=300, save_as = None, name = None, sample_points = None):
    def TN_density(x, scaling_constant, A, b):
        mu = np.zeros(x.shape[0])
        cov = np.eye(x.shape[0])
        base = np.dot(np.linalg.inv(A), (x - b))
        if np.linalg.norm(base) <= 1:
            return multivariate_normal.pdf(base, mean=mu, cov=cov) * scaling_constant
        else:
            return 0

    # Generate meshgrid
    x = np.linspace(xrange[0], xrange[1], grid_size)
    y = np.linspace(yrange[0], yrange[1], grid_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            Z[i, j] = TN_density(np.array([X[i, j], Y[i, j]]), scaling_constant, A, b)

    # Plot contours
    plt.contourf(X, Y, Z, cmap='BuGn')
    plt.colorbar(label='Density')
    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('{}'.format(name))  # LaTeX syntax for the title

    if sample_points is not None:
        plt.scatter(sample_points[:, 0], sample_points[:, 1], color='red', label='Sample Points', s=10)

    if save_as is not None:
        plt.savefig(save_as)
        plt.clf()
    else:
        plt.show() # Clear the current figure
    # plt.show()

# scaling_constant= 1 / quad(gaussian_density, -1, 1, args=(0, 1))[0] # args=(mean, sigma)   
# gaussian_info = read_data(pathname = "gaussian_input/iter_data", filename = "info.json")
# dim = gaussian_info['Dimension']
# K = gaussian_info['Number of measures']
# n_samples = 2000
# radius_truncate = 1
# nu_para = []
# for i in range(K):
#     mean = np.array(gaussian_info['Mean of nu_{}'.format(i)])
#     A_matrix = np.array(gaussian_info['A_matrix of nu_{}'.format(i)])
#     nu_para.append((mean, A_matrix))

# nu_list = []
# for i in range(K):
#     nu_list.append(Measure(dim, "gaussian", [nu_para[i]]))
    
# for k in range(K):
#     b, A = nu_list[k].parameters[0][0], nu_list[k].parameters[0][1]
#     plot_TN_density_surface(scaling_constant, A, b, save_as = f"gaussian_input/measure_visualization/nu_{k}.png", name = f"$\\nu_{k + 1}$")


# b_bary = np.array(read_data(pathname = "gaussian_input/FP_data", filename = "barycenter_info.json")["b of barycenter"])
# A_bary = np.array(read_data(pathname = "gaussian_input/FP_data", filename = "barycenter_info.json")["A of barycenter"])
# sample_points = np.array(read_data(pathname = "gaussian_input/iter_data2", filename = "barycenter.json"))
# plot_TN_density_surface(scaling_constant, A_bary, b_bary, save_as = "gaussian_input/measure_visualization/barycenter_GT.png", name = "the barycenter", sample_points=None)
# plot_TN_density_surface(scaling_constant, A_bary, b_bary, save_as = "gaussian_input/measure_visualization/barycenter.png", name = "the barycenter", sample_points=sample_points)

bary_V = read_data(pathname = "gaussian_input/FP_data", filename = "solve_data.json")["Objective value"]
bary_V_iter = list(read_data(pathname = "gaussian_input/iter_data2", filename = "iter_objective.json").values())


plt.plot(range(1, len(bary_V_iter) + 1), bary_V_iter, marker='o', linestyle='-', color='red', label='iterative output measure')
plt.axhline(y=bary_V, color='green', linestyle='--', label='the true barycenter')

# Add labels and title
plt.xlabel('Iterations')
plt.ylabel('V-value')
plt.title('Convergence of the proposed algorithm to the barycenter')
plt.legend()

# Add text annotations for each point
for i, v in enumerate(bary_V_iter):
    plt.text(i + 1, v, f'{v:.2f}', ha='right', va='bottom', color = 'red')  # Format to two decimal places

# Add text annotation for the constant line
plt.text(1, bary_V, f'{bary_V:.2f}', ha='right', va='bottom', color = 'green')  # Format to two decimal places

# Display the plot
plt.grid(True)
plt.show()
import sys, os
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from Iterative_Scheme import * 
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd

def inside_ball(x, y, radius=1):
    return x**2 + y**2 <= radius**2

# Perform the double integral over the unit ball
# def multivariate_ball_integral(mean, covariance_matrix, truncate_radius):
#     integrand = lambda x, y: multivariate_normal.pdf([x, y], mean, covariance_matrix)
#     integral, _ = dblquad(integrand, -truncate_radius, truncate_radius, lambda x: -np.sqrt(truncate_radius**2 - x**2), lambda x: np.sqrt(truncate_radius**2 - x**2))
#     return integral

def multivariate_ball_integral(dim, mixture, truncate_radius):
    mixture_measure = Measure(dim, "mixture_gaussian", mixture)
    samples = mixture_measure.generate_mixture_gaussian_sample(size=10000)
    return sum(inside_ball(x, y, truncate_radius) for x, y in samples) / len(samples)

def gaussian_density(x, mu, sigma):
    return 1 / (np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

# def multivariate_gaussian_density(x, mu, cov):
#     k = len(mu) - 1
#     norm_factor = (2 * np.pi) ** (-k - 1 / 2) * np.linalg.det(cov) ** (-0.5)
#     cov_inv = np.linalg.inv(cov)
#     exponent = -0.5 * sum((x[i] - mu[i]) * cov_inv[i, j] * (x[j] - mu[j]) for i in range(k) for j in range(k))
#     return norm_factor * np.exp(exponent)


# Define the integrand function
# def truncated_density(x, y, mixture, truncate_radius, scaling_factor):
#     density = 0
#     for mean, covariance_matrix in mixture:
#         density += multivariate_normal.pdf([x, y], mean=mean, cov=covariance_matrix) * inside_ball(x, y, truncate_radius) * scaling_factor
#     return density / len(mixture)

def truncated_density(x, y, mixture, truncate_radius, scaling_factor):
    density = 0
    for mean, covariance_matrix in mixture:
        density += multivariate_normal.pdf([x, y], mean=mean, cov=covariance_matrix) * inside_ball(x, y, truncate_radius) * scaling_factor
    return density / len(mixture)

def scaling_factor(mixture, truncate_radius):
    scaling_factor = 1 / multivariate_ball_integral(2, mixture, truncate_radius)
    return scaling_factor

def plot_truncated_density(mixture, truncate_radius, xrange=(-8, 8), yrange=(-8, 8), grid_size=400, save_as = None, name = None, sample_points = None):
    scaling = scaling_factor(mixture, truncate_radius)
    # Generate meshgrid
    x = np.linspace(xrange[0], xrange[1], grid_size)
    y = np.linspace(yrange[0], yrange[1], grid_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            Z[i, j] = truncated_density(X[i, j], Y[i, j], mixture, truncate_radius, scaling)
            # breakpoint()

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

mixture_gaussian_info = read_data(pathname = "mixture_input/iter_data", filename = "info.json")
dim = mixture_gaussian_info['Dimension']
K = mixture_gaussian_info['Number of measures']
truncate_radius = mixture_gaussian_info['Radius of truncation']
n_mix = mixture_gaussian_info['Number of mixture components']
all_mixture = []
for k in range(K):
    mixture = []
    for i in range(n_mix):
        mean = np.array(mixture_gaussian_info['mixture components of nu_{}'.format(k)][i][0])
        covariance_matrix = np.array(mixture_gaussian_info['mixture components of nu_{}'.format(k)][i][1])
        mixture.append((mean, covariance_matrix))
    all_mixture.append(mixture)

for i in range(K):
    plot_truncated_density(all_mixture[i], truncate_radius, save_as= "mixture_input/measure_visualization/nu_{}.png".format(i), name = "$\\nu_{}$".format(i + 1))
    
barycenter = np.array(read_data(pathname = "mixture_input/iter_data", filename = "barycenter.json"))
barycenter_df = pd.DataFrame(barycenter, columns=["x", "y"])
plt.figure(figsize=(8, 6))
sns.kdeplot(data=barycenter_df, x="x", y="y", cmap = "BuGn", fill=True, bw_adjust=0.5, n_levels=20)
sns.scatterplot(data=barycenter_df, x="x", y="y", color='red', label='Sample Points', s=10)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kernel density estimate of the barycenter with sample points')
plt.legend()
# Set x-axis and y-axis limits
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.savefig("mixture_input/measure_visualization/barycenter.png")
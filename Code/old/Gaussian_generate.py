import numpy as np

## Location-Scatter Families.

# Generate random Gaussian distributions
# INPUT: 
# K - number of Gaussian distributions to generate;
# dim - dimensionality of the Gaussian distributions;
# seed - random seed for reproducibility.
# OUTPUT:
# gaussians - list of tuples, where each tuple contains the mean vector and covariance matrix of a Gaussian distribution.
def generate_gaussians(K, dim, seed=None):
    np.random.seed(seed)
    gaussians = []
    for _ in range(K):
        # Generate random mean vector
        mean = np.random.uniform(low=-10, high=10, size=(dim,))
        
        # Generate random covariance matrix
        A = np.random.randn(dim, dim)
        covariance_matrix = np.dot(A, A.T)  # Ensures positive semi-definite covariance matrix
        
        gaussians.append((mean, covariance_matrix))
    
    return gaussians


# Generate random vectors from Gaussian distributions
# INPUT:
# gaussians - list of tuples, where each tuple contains the mean vector and covariance matrix of a Gaussian distribution;
# N - number of random vectors to generate;
# seed - random seed for reproducibility.
# OUTPUT:
# random_vectors - list of random vectors generated from the Gaussian distributions.


def generate_gaussian_vectors(gaussian, N, seed=None):
    np.random.seed(seed)
    random_vectors = []
    for _ in range(N):
        (mean, covariance_matrix) = gaussian
        # Generate random vector from selected Gaussian distribution
        vector = np.random.multivariate_normal(mean, covariance_matrix)
        random_vectors.append(vector)
    
    return np.array(random_vectors) # sample.shape: (N, dim) 

# rho0 = generate_gaussians(1, 3, seed=38)[0]
# sample = generate_gaussian_vectors(rho0, 2, seed=38)
# print(rho0)
# print(sample)
# breakpoint()






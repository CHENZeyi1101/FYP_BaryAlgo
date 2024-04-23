import ndtest
import numpy as np
import sys, os
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by appending ".." to the current directory
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
# Add the parent directory to the Python module search path
sys.path.append(parent_dir)

from Iterative_Scheme import * 


barycenter_iter = np.array(read_data(pathname = "gaussian_input/iter_data2", filename = "barycenter.json"))
barycenter_FP = np.array(read_data(pathname = "gaussian_input/FP_data", filename = "barycenter_sample.json"))

x_1 = np.array(barycenter_iter[:, 0])
y_1 = np.array(barycenter_iter[:, 1])
x_2 = np.array(barycenter_FP[:, 0])
y_2 = np.array(barycenter_FP[:, 1])

P, D = ndtest.ks2d2s(x_1, y_1, x_2, y_2, extra=True)
print(f"{P=:.3g}, {D=:.3g}")

# # Scatter plot for barycenter_iter
plt.scatter(barycenter_iter[:, 0], barycenter_iter[:, 1], label='barycenter_iter', color='red', marker='o')

# Scatter plot for barycenter_FP
plt.scatter(barycenter_FP[:, 0], barycenter_FP[:, 1], label='barycenter_FP', color='green', marker='x')

# Add labels and legend
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Scatter Plot of Two Data Sets')
plt.legend()

# Show plot
plt.show()




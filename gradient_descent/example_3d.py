import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import gradient_descent as gd
import numpy as np
from math import log
from matplotlib import cm # color map

# Make some data.
x_values = np.linspace(start=-2, stop=2, num=200)
y_values = np.linspace(start=-2, stop=2, num=200)
x_values, y_values = np.meshgrid(x_values, y_values) # Convert to 2-d array

def f(x, y):
    r = 3**(-x**2 - y**2)
    return 1 / (r + 1)

# Partial derivative functions.
def fpx(x, y):
    r = 3**(-x**2 - y**2)
    return 2*x*log(3)*r / (r + 1)**2

def fpy(x, y):
    r = 3**(-x**2 - y**2)
    return 2*y*log(3)*r / (r + 1)**2

# Calling gradient descent function.
local_mins, values_array, gradients_array = gd.gradient_descent_3d(function=f, initial_guess=np.array([1.8, 1.0]), multiplier=0.1)

# Generating 3D Plot
fig = plt.figure(figsize=[16, 12])
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('Y', fontsize=20)
ax.set_zlabel('f(x, y) - Cost', fontsize=20)

ax.plot_surface(x_values, y_values, f(x_values, y_values), cmap=cm.coolwarm, alpha=0.4)
ax.scatter(values_array[:, 0], values_array[:, 1], 
           f(values_array[:, 0], values_array[:, 1]), s=50, color='red')

plt.show()

import matplotlib.pyplot as plt
import numpy as np
import gradient_descent as gd

# Make some data.
x_values = np.linspace(-2, 2, 1000)

def f(x):
    return x**4 - 4*x**2 + 5

def df(x):
    return 4*x**3 - 8*x

# Calling gradient descent function.
local_min, value_x, gradient_list = gd.gradient_descent_2d(function=f, initial_guess=0.5)

def displayGraph(scatterGradientDescend):
    plt.figure(figsize=[15, 5])

    # 1 Chart: Cost function.
    plt.subplot(1, 2, 1)

    plt.title('Cost function', fontsize=17)
    plt.xlabel('X', fontsize=16)
    plt.ylabel('g(x)', fontsize=16)
    plt.xlim(-2, 2)
    plt.ylim(0.5, 5.5)

    plt.plot(x_values, f(x_values), color='blue', linewidth=3)
    if scatterGradientDescend == True:
        plt.scatter(value_x, f(np.array(value_x)), color='red', s=100, alpha=0.6)

    # 2 Chart: Derivative.
    plt.subplot(1, 2, 2)

    plt.title('Slope of the cost function', fontsize=17)
    plt.xlabel('X', fontsize=16)
    plt.ylabel('dg(x)', fontsize=16)
    plt.xlim(-2, 2)
    plt.ylim(-6, 8)
    plt.grid()


    plt.plot(x_values, df(x_values), color='skyblue', linewidth=5)
    if scatterGradientDescend == True:
        plt.scatter(value_x, gradient_list, color='red', s=100, alpha=0.5)

    plt.show()

displayGraph(scatterGradientDescend=True)
print('Local min occurs at: ', local_min)
print('Cost at this minimum is: ', f(local_min))
print('Number of steps: ', len(value_x))

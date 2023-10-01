import numpy as np
from sympy import symbols, diff

def gradient_descent_2d(function, initial_guess, multiplier=0.02, precision=0.001, max_iter=300):
    a = symbols('x') # Set x equal to a for simplicity.

    new_param = initial_guess
    value_list = [new_param] # X list
    gradient_list = [diff(function(a), a).evalf(subs={a:new_param})] # Slope list

    for n in range(max_iter):
        previous_param = new_param

        # Calculate the derivative of the function with respect to x and find its value at the current parameter.
        # The steeper the slope, the greater the distance is from the minimum value.
        gradient = diff(function(a), a).evalf(subs={a:previous_param})

        # new_parameter = old_parameter - learning_rate * gradient
        new_param = previous_param - multiplier * gradient

        # The smaller this variable, the closer we are to the minimum value.
        step_size = abs(new_param - previous_param)

        value_list.append(new_param)
        gradient_list.append(gradient)

        if step_size < precision:
            break

    return new_param, value_list, gradient_list


def gradient_descent_3d(function, initial_guess, multiplier=0.02, precision=0.001, max_iter=300):
    # In a 3D graph, we deal with two slopes (x and y) instead of just one.

    a, b = symbols('x, y') # Set x and y equal to a and b for simplicity.

    new_params = initial_guess

    # Arrange x values in one dimension and the y values in another.
    values_array = new_params.reshape(1, 2)

    # Slope array
    gradient_x = diff(function(a, b), a).evalf(subs={a:new_params[0], b:new_params[1]})
    gradient_y = diff(function(a, b), b).evalf(subs={a:new_params[0], b:new_params[1]})
    gradients_array = np.array([gradient_x, gradient_y])

    for n in range(max_iter):
        previous_params = new_params

        # Calculate the derivative of the function with respect to x and y and find its value at the current parameters.
        # The steeper the slope, the greater the distance is from the minimum value.
        gradient_x = diff(function(a, b), a).evalf(subs={a:previous_params[0], b:previous_params[1]})
        gradient_y = diff(function(a, b), b).evalf(subs={a:previous_params[0], b:previous_params[1]})
        gradients_array = np.array([gradient_x, gradient_y])


        # new_parameter = old_parameter - learning_rate * gradient
        new_params = previous_params - multiplier * gradients_array

        # The smaller this variable, the closer we are to the minimum value.
        # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        step_size = abs(new_params - previous_params).all()

        values_array = np.concatenate((values_array, new_params.reshape(1, 2)), axis=0)

        if step_size < precision:
            break

    return new_params, values_array, gradients_array



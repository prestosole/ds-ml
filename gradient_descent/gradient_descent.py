def gradient_descent(derivative_func, initial_guess, multiplier=0.02, precision=0.001, max_iter=300):
    new_x = initial_guess
    x_list = [new_x]
    slope_list = [derivative_func(new_x)]

    for n in range(max_iter):
        current_x = new_x
        gradient = derivative_func(current_x) # slope at current_x; The steeper the slope, the less accurate the result will be
        new_x = current_x - multiplier * gradient # the main function of the gradient descent

        step_size = abs(new_x - current_x) # the lower the value, the more accurate the result will be
        x_list.append(new_x)
        slope_list.append(derivative_func(new_x))

        if step_size < precision:
            break

    return new_x, x_list, slope_list # local min and points for scattered plot

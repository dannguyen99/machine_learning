from random import sample
import numpy as np
import matplotlib.pyplot as plt

independent_variable = np.array(
    [[1, 30, 3, 6], [1, 43, 4, 8], [1, 25, 2, 3], [1, 51, 4, 9], [1, 40, 3, 5], [1, 20, 1, 2]])
dependent_variable = np.array([[2.5], [3.4], [1.8], [4.5], [3.2], [1.6]])
theta = [0] * 4
rate = 0.001
number_iter = 100
b_size = 2


def cal_derivative(x_values, y_values, thetas, position):
    sigma = 0
    for i in range(len(y_values)):
        ht = 0
        for j in range(len(thetas)):
            ht += thetas[j] * x_values[i][j]
        ht -= y_values[i]
        sigma += ht * x_values[i][position]
    return sigma / len(y_values)


def cal_next_theta(x_values, y_values, thetas, learning_rate):
    new_thetas = []
    for i in range(len(thetas)):
        new_theta = thetas[i] - learning_rate * cal_derivative(x_values, y_values, thetas, i)
        new_thetas.append(new_theta)
    return new_thetas


def cal_cost(x_values, y_values, thetas):
    sigma = 0
    for i in range(len(y_values)):
        ht = 0
        for j in range(len(thetas)):
            ht += thetas[j] * x_values[i][j]
        ht -= y_values[i]
        sigma += ht ** 2
    return sigma / (2 * len(y_values))


def mini(x_values, y_values, thetas, batch_size, learning_rate=0.001, no_iter=1000):
    print("running with inputs")
    print("x values are", x_values)
    print("y values are", y_values)
    min_cost = cal_cost(x_values, y_values, thetas)
    best_thetas = thetas
    if batch_size > len(x_values):
        return "invalid, batch size must >= whole batch size"
    x_axis = []
    y_axis = []
    for i in range(no_iter):
        x_axis.append(i)
        y_axis.append(cal_cost(x_values, y_values, thetas))
        samples = sample(range(0, len(x_values)), batch_size)
        x_input = []
        y_input = []
        for s in samples:
            x_input.append(x_values[s])
            y_input.append(y_values[s])
        # print("x is", x_input, "and y is", y_input)
        thetas = cal_next_theta(x_input, y_input, thetas, learning_rate)
        if cal_cost(x_values, y_values, thetas) < min_cost:
            best_thetas = thetas
            min_cost = cal_cost(x_values, y_values, thetas)
    print("best thetas is", best_thetas)
    print("minimum cost is ", min_cost)
    plt.plot(x_axis, y_axis)
    plt.xlabel("epochs")
    plt.ylabel("value of cost function")
    plt.show()
    return best_thetas, min_cost


mini(independent_variable, dependent_variable, theta, b_size, rate, number_iter)

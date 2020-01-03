import numpy as np
from numpy.linalg import inv

independent_variable = np.array(
    [[1, 30, 3, 6], [1, 43, 4, 8], [1, 25, 2, 3], [1, 51, 4, 9], [1, 40, 3, 5], [1, 20, 1, 2]])
dependent_variable = np.array([[2.5], [3.4], [1.8], [4.5], [3.2], [1.6]])
theta = [0] * 4
rate = 0.001
number_iter = 1000


def cal_thetas(x_values, y_values):
    return (inv(x_values.T.dot(x_values)).dot(x_values.T)).dot(y_values)


def cal_cost(x_values, y_values, thetas):
    sigma = 0
    for i in range(len(y_values)):
        ht = 0
        for j in range(len(thetas)):
            ht += thetas[j] * x_values[i][j]
        ht -= y_values[i]
        sigma += ht ** 2
    return sigma / (2 * len(y_values))


def normal_equation(x_values, y_values):
    print("running with inputs")
    print("x values are", x_values)
    print("y values are", y_values)
    best_thetas = cal_thetas(x_values, y_values)
    min_cost = cal_cost(x_values, y_values, best_thetas)
    print("best thetas is", best_thetas)
    print("minimum cost is ", min_cost)
    return best_thetas, min_cost


normal_equation(independent_variable, dependent_variable)

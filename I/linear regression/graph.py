from batch_learning import batch_learning
from mini_batch import mini
from stochastic_batch import stochastic
import numpy as np
import matplotlib.pyplot as plt

independent_variable = np.array(
    [[1, 30, 3, 6], [1, 43, 4, 8], [1, 25, 2, 3], [1, 51, 4, 9], [1, 40, 3, 5], [1, 20, 1, 2]])
dependent_variable = np.array([[2.5], [3.4], [1.8], [4.5], [3.2], [1.6]])
theta = [0] * 4
rate = 0.001
number_iter = 100
b_size = 3

batch_learning(independent_variable, dependent_variable, theta,rate, number_iter)
mini(independent_variable, dependent_variable, theta, b_size, rate, number_iter)
stochastic(independent_variable, dependent_variable, theta, rate, number_iter)
plt.legend(['batch learning', 'mini-batch', 'stochastic batch'])
plt.show()
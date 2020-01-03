from math import exp


def calculate_with_theta(thetas, x):
    sig = 0
    for i in range(len(thetas)):
        sig -= thetas[i] * (x ** i)
    return 1 / (1 + exp(sig))


x_value = [2.5, 3.5, 5.6, 2.2, 6.9, 9.6]
y_value = [0, 0, 1, 0, 1, 1]
for i in x_value:
    print("with x is: ", i, "the  value is ", calculate_with_theta([0.5, 0.7], i))


# print(calculate_with_theta([0.5,0.7], 2.5))
# print(calculate_with_theta([0,0], 3.5))

def cal_next_theta(thetas, x_value, y_value, learning_rate=0.001):
    sig = 0
    new_thetas = []
    for i in range(len(x_value)):
        sig += (calculate_with_theta(thetas, x_value[i]) - y_value[i]) * x_value[i]
    sig = sig / len(x_value)
    for i in thetas:
        new_thetas.append(i - learning_rate * sig)
    return new_thetas


print(cal_next_theta([0, 0], x_value, y_value))
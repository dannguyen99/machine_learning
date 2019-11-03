import matplotlib.pyplot as plt

base = [1] * 6
size = [30, 43, 25, 51, 40, 20]
no_floors = [3, 4, 2, 4, 3, 1]
no_rooms = [6, 8, 3, 9, 5, 2]
price = [2.5, 3.4, 1.8, 4.5, 3.2, 1.6]
theta = [0] * 4
rate = 0.001
plt.plot(size, price, 'ro')
plt.xlabel("size(m^2)")
plt.ylabel("price(b.VND)")


def cal_derivative(x_values, y_values, thetas, position):
    sigma = 0
    for i in range(len(y_values)):
        sum = 0
        for j in range(len(thetas)):
            sum += thetas[j] * x_values[j][i]
        sum -= y_values[i]
        sigma += sum * x_values[position][i]
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
        sum = 0
        for j in range(len(thetas)):
            sum += thetas[j] * x_values[j][i]
        sum -= y_values[i]
        sigma += sum ** 2
    return sigma / (2 * len(y_values))


temp = theta
min_cost = cal_cost([base, size, no_floors, no_rooms], price, temp)
best_thetas = theta
for i in range(1000):
    # print(cal_next_theta([base, size, no_floors, no_rooms], price, temp, rate))
    # print(cal_cost([base, size, no_floors, no_rooms], price, temp))
    temp = cal_next_theta([base, size, no_floors, no_rooms], price, temp, rate)
    if cal_cost([base, size, no_floors, no_rooms], price, temp) < min_cost:
        min_cost = cal_cost([base, size, no_floors, no_rooms], price, temp)
        best_thetas = temp

print(min_cost)

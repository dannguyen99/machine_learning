import matplotlib.pyplot as plt

size = [30, 43, 25, 51, 40, 20]
price = [2.5, 3.4, 1.8, 4.5, 3.2, 1.6]
plt.plot(size, price, 'ro')
plt.xlabel("size(m^2)")
plt.ylabel("price(b.VND)")


# plt.show()


def cal_next_a(a, b, x_value, y_value, learning_rate=0.0001):
    sigma = 0
    for i in range(len(x_value)):
        # print("x is: ", x_value[i], " and y is: ", y_value[i]
        sigma += (a * x_value[i] + b - y_value[i]) * x_value[i]
    a1 = a - learning_rate * sigma / len(x_value)
    return a1


def cal_next_b(a, b, x_value, y_value, learning_rate=0.0001):
    sigma = 0
    for i in range(len(x_value)):
        sigma += (a * x_value[i] + b - y_value[i])
    b1 = b - learning_rate * sigma / len(x_value)
    return b1


# print("a1 is ",cal_next_a(0, 0, size, price))
# print("b1 is ",cal_next_b(0, 0, size, price))
def cost(a, b, x_value, y_value):
    sigma = 0
    for i in range(len(x_value)):
        sigma += (a * x_value[i] + b - y_value[i])**2
    return sigma / (2 * len(x_value))


def cal(a, b):
    y = []
    for i in range(60):
        y.append(a * i + b)
    return y


a = 0
b = 0
temp = a
min = cost(a, b, size, price)
for i in range(100):
    # print("error is ", cost(a, b, size, price))
    a = cal_next_a(a, b, size, price)
    # print("a", i, " is ", a)
    b = cal_next_b(temp, b, size, price)
    # print("b", i, " is ", b)
    temp = a
    if abs(cost(a, b, size, price)) < abs(min):
        min = cost(a, b, size, price)
        best_a = a
        best_b = b
    plt.plot(range(60), cal(a,b))
    print()

print("minimum error is ", min)
print("best a is", best_a, "and best b is", best_b)
plt.plot(range(60),cal(best_a,best_b))
# plt.plot(range(60),cal(1, 1))
plt.show()
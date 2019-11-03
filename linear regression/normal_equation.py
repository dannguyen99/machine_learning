import numpy
from numpy.linalg import inv

size = numpy.array([[30], [43], [25], [51], [40], [20]])
price = numpy.array([[2.5], [3.4], [1.8], [4.5], [3.2], [1.6]])
# y = inv(numpy.multiply(size.T,size))
y = numpy.multiply(size.T, size)
print(numpy.linalg.det(y))
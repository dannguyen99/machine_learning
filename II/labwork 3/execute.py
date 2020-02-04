import numpy as np
from sklearn import datasets, metrics


class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

    def labels(self, inputs):
        result = []
        for i in inputs:
            result.append(self.predict(i))
        return result


def main():
    iris = datasets.load_iris()
    training_inputs = iris.data[:100]

    labels = iris.target

    perceptron = Perceptron(4, threshold=5000)
    perceptron.train(training_inputs, labels)

    print(labels[:100])
    print(perceptron.labels(training_inputs))


if __name__ == '__main__':
    main()

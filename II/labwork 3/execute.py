import numpy as np
from sklearn import datasets, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA, TruncatedSVD
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns


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


def visualize_perceptron(dataset):
    X = dataset.data

    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    Y = dataset.target
    Y[Y == 2] = 1

    perceptron = Perceptron(2, threshold=300)
    perceptron.train(X, Y)

    colors = ['r', 'g', 'b']
    for i, color in zip(range(2), colors):
        plt.scatter(X[Y == i, 0], X[Y == i, 1], s=50, marker='o', color=color)
    plt.xlabel("Principle Component 1")
    plt.ylabel("Principle Component 2")

    # print(X.T)
    print(perceptron.labels(X))

    weights = perceptron.weights
    slope = - weights[1] / weights[2]
    x2cut = -weights[0] / weights[2]
    min_val = min(X.T[0])
    max_val = max(X.T[0])
    x1 = [min_val, max_val]
    x2 = [slope * min_val + x2cut, slope * max_val + x2cut]
    plt.plot(x1, x2)
    plt.title("Perceptron classifier for Breast Cancer Dataset")


def visualize_knn(dataset):
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X, y)
    y_pred = knn.predict(X_test)
    ACC = metrics.accuracy_score(y_test, y_pred)
    print(ACC)

    colors = ['r', 'g', 'b']
    for i, color in zip(range(3), colors):
        plt.scatter(X[y == i, 0], X[y == i, 1], s=50, marker='o', color=color)
    plt.xlabel("Principle Component 1")
    plt.ylabel("Principle Component 2")


def evaluate_k_model(dataset, dataset_name, n_neighbors=5, is_normalize=False, use_PCA=False, is_print=True):
    results = {}
    X = dataset.data
    y = dataset.target
    if is_normalize:
        X = normalize(X)
    if use_PCA:
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
    model.fit(X_train, y_train)
    labels_true = y_test
    labels_pred = model.predict(X_test)
    ACC = metrics.accuracy_score(labels_true, labels_pred)
    results['ACC'] = ACC
    ARI = metrics.adjusted_rand_score(labels_true, labels_pred)
    results['ARI'] = ARI
    MI = metrics.mutual_info_score(labels_true, labels_pred)
    results['MI'] = MI
    NMI = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    results['NMI'] = NMI
    JC = metrics.jaccard_score(labels_true, labels_pred, average=None)
    results['JC'] = JC
    if is_print:
        print("Evaluating", dataset_name, "dataset")
        print("Accuracy =", ACC)
        print("Adjusted Rand index =", ARI)
        print("Mutual Information =", MI)
        print("Normalized Mutual Information =", NMI)
        print("Jaccard Score = ", JC)
        print()
    return results


def evaluate_number_k(dataset, dataset_name, is_normalize=False, use_PCA=False):
    k_range = range(1, 26)
    scores = {}
    scores_list = []
    X = dataset.data
    if is_normalize:
        X = normalize(X)
    if use_PCA:
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test, y_pred)
        scores_list.append(metrics.accuracy_score(y_test, y_pred))
    plt.figure()
    plt.plot(k_range, scores_list)
    plt.title(dataset_name)
    plt.xlabel("Value of K for KNN".format(dataset_name))
    plt.ylabel("Testing Accuracy")


def main():
    iris = datasets.load_iris()
    wine = datasets.load_wine()
    breast_cancer = datasets.load_breast_cancer()
    # visualize_perceptron(breast_cancer)
    visualize_knn(iris)
    evaluate_k_model(dataset=iris, dataset_name="Iris", n_neighbors=5, is_normalize=False, use_PCA=True)
    # evaluate_number_k(dataset=iris, dataset_name="Iris", is_normalize=True)
    plt.show()


if __name__ == '__main__':
    main()

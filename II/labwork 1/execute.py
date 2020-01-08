import math

import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, datafile_path, names, compute_feature):
        self.data = pd.read_csv(datafile_path, sep=',', skipinitialspace=True, names=names)
        self.feature_size = compute_feature
        self.x = self.data.iloc[:, 0:self.feature_size].values
        self.y = self.data.iloc[:, self.feature_size].values

    @staticmethod
    def mean(serie):
        return round(sum(serie) / len(serie), 4)

    def mean_all(self, is_print=False):
        mean_values = []
        if is_print:
            print("attribute \t\t mean")
        for i in self.data:
            column = self.data[i]
            if column.dtype == 'int64' or column.dtype == 'float64':
                mean_value = self.mean(column)
                if is_print:
                    print(i, "\t", mean_value)
                mean_values.append(mean_value)
        return mean_values

    def standard_deviation(self, serie):
        return math.sqrt(self.variance(serie))

    def variance(self, serie):
        mean_value = self.mean(serie)
        sum_var = 0
        for i in serie:
            sum_var += (mean_value - i) ** 2
        return round(sum_var / len(serie), 4)

    def variance_all(self, is_print=False):
        variance_values = []
        if is_print:
            print("attribute \t\t variance")
        for i in self.data:
            column = self.data[i]
            if column.dtype == 'int64' or column.dtype == 'float64':
                variance_value = self.variance(column)
                if is_print:
                    print(i, "\t", variance_value)
                variance_values.append(variance_value)
        return variance_values

    def covariance(self, serie_x, serie_y):
        mean_x = self.mean(serie_x)
        mean_y = self.mean(serie_y)
        sum_covariance = 0
        for x, y in zip(serie_x, serie_y):
            sum_covariance += (x - mean_x) * (y - mean_y)
        return round(sum_covariance / len(self.data.index), 3)

    def covariance_all(self, is_print=False):
        covariance_values = np.zeros((self.feature_size, self.feature_size))
        if is_print:
            print("Here is the covariance matrix")
            print("attribute\t\t", end="")
        for i in self.data:
            column = self.data[i]
            if column.dtype == 'int64' or column.dtype == 'float64':
                if is_print:
                    print(i, "\t", end="")
        print()
        for i, a in zip(self.data, range(self.feature_size)):
            column = self.data[i]
            if column.dtype == 'int64' or column.dtype == 'float64':
                if is_print:
                    print(i, "\t", end="")
                for j, b in zip(self.data, range(self.feature_size)):
                    row = self.data[j]
                    if row.dtype == 'int64' or row.dtype == 'float64':
                        covariance_value = self.covariance(column, row)
                        if is_print:
                            print(covariance_value, "\t\t\t", end="")
                        covariance_values[a][b] = covariance_value
                if is_print:
                    print()
        return covariance_values

    def correlation(self, serie_x, serie_y):
        std_x = self.standard_deviation(serie_x)
        std_y = self.standard_deviation(serie_y)
        return round(self.covariance(serie_x, serie_y) / (std_x * std_y), 3)

    def correlation_all(self, is_print=False):
        correlation_values = np.zeros((self.feature_size, self.feature_size))
        if is_print:
            print("Here is the covariance matrix")
            print("attribute\t\t", end="")
        for i in self.data:
            column = self.data[i]
            if column.dtype == 'int64' or column.dtype == 'float64':
                if is_print:
                    print(i, "\t", end="")
        print()
        for i, a in zip(self.data, range(self.feature_size)):
            column = self.data[i]
            if column.dtype == 'int64' or column.dtype == 'float64':
                if is_print:
                    print(i, "\t", end="")
                for j, b in zip(self.data, range(self.feature_size)):
                    row = self.data[j]
                    if row.dtype == 'int64' or row.dtype == 'float64':
                        correlation_value = self.correlation(column, row)
                        if is_print:
                            print(correlation_value, "\t\t\t", end="")
                        correlation_values[a][b] = correlation_value
                if is_print:
                    print()
        return correlation_values

    def pca(self, is_print=False):
        eig_vals, eig_vecs = np.linalg.eig(self.correlation_all())
        for ev in eig_vecs:
            np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
        # Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort()
        eig_pairs.reverse()

        matrix_w = np.hstack((eig_pairs[0][1].reshape(4, 1),
                              eig_pairs[1][1].reshape(4, 1)))
        # print(self.data.values)
        Y = self.x.dot(matrix_w)
        principalDf = pd.DataFrame(data=Y
                                   , columns=['principal component 1', 'principal component 2'])
        if is_print:
            print(principalDf)
        return Y


d = Dataset('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
            ['sepal length', 'sepal width', 'petal length', 'petal width', 'class'], 4)
d.mean_all(is_print=True)
d.variance_all(is_print=True)
d.covariance_all(is_print=True)
d.correlation_all(is_print=True)
d.pca(is_print=True)

import math

import pandas as pd
from sklearn.preprocessing import StandardScaler

# study dataset properties
abalone_dataset = pd.read_csv('abalone.csv', sep=',', skipinitialspace=True)
iris_dataset = pd.read_csv('iris.csv', sep=',', skipinitialspace=True)

print(iris_dataset.corr())


# print(abalone_dataset.mean())
# print(abalone_dataset.var())
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(abalone_dataset.corr())
# print(abalone_dataset.cov())
# print(abalone_dataset.corr())
# print(iris_dataset.describe())
class Dataset:
    def __init__(self, datafile_path):
        self.features_name = None
        self.target_name = None
        self.data = pd.read_csv(datafile_path, sep=',', skipinitialspace=True)

    def set_features_name(self, features):
        self.features_name = features

    def set_target_name(self, target):
        self.target_name = target

    def standardize_data(self):
        if self.features_name is None or self.target_name is None:
            pass
        y = self.data.loc[:, self.target_name].values
        StandardScaler().fit_transform(self.data.loc[:, self.features_name].values)

    @staticmethod
    def mean(serie):
        return round(sum(serie) / len(serie), 4)

    def mean_all(self):
        mean_values = []
        print("attribute \t\t mean")
        for i in self.data:
            column = self.data[i]
            if column.dtype == 'int64' or column.dtype == 'float64':
                mean_value = self.mean(column)
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

    def variance_all(self):
        variance_values = []
        print("attribute \t\t variance")
        for i in self.data:
            column = self.data[i]
            if column.dtype == 'int64' or column.dtype == 'float64':
                variance_value = self.variance(column)
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

    def covariance_all(self):
        covariance_values = []
        print("attribute\t\t", end="")
        for i in self.data:
            column = self.data[i]
            if column.dtype == 'int64' or column.dtype == 'float64':
                print(i, "\t", end="")
        print()
        for i in self.data:
            column = self.data[i]
            if column.dtype == 'int64' or column.dtype == 'float64':
                print(i, "\t", end="")
                for j in self.data:
                    row = self.data[j]
                    if row.dtype == 'int64' or row.dtype == 'float64':
                        covariance_value = self.covariance(column, row)
                        print(covariance_value, "\t\t\t", end="")
                        covariance_values.append(covariance_value)
                print()
        return covariance_values

    def correlation(self, serie_x, serie_y):
        std_x = self.standard_deviation(serie_x)
        std_y = self.standard_deviation(serie_y)
        return round(self.covariance(serie_x, serie_y) / (std_x * std_y), 3)

    def correlation_all(self):
        correlation_values = []
        print("attribute\t\t", end="")
        for i in self.data:
            column = self.data[i]
            if column.dtype == 'int64' or column.dtype == 'float64':
                print(i, "\t", end="")
        print()
        for i in self.data:
            column = self.data[i]
            if column.dtype == 'int64' or column.dtype == 'float64':
                print(i, "\t", end="")
                for j in self.data:
                    row = self.data[j]
                    if row.dtype == 'int64' or row.dtype == 'float64':
                        correlation_value = self.correlation(column, row)
                        print(correlation_value, "\t\t\t", end="")
                        correlation_values.append(correlation_value)
                print()
        return correlation_values


d = Dataset('abalone.csv')
print(d.data.describe())
print(d.data.cov())
d.covariance_all()
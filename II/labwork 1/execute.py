import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# study dataset properties
abalone_dataset = pd.read_csv('abalone.csv', sep=',')
iris_dataset = pd.read_csv('iris.csv', sep=',')
# print(abalone_dataset.mean())
# print(abalone_dataset.var())
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(abalone_dataset.corr())
# print(abalone_dataset.cov())
# print(abalone_dataset.corr())
# print(iris_dataset.describe())



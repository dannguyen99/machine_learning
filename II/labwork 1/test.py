import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
# load dataset into Pandas DataFrame
df = pd.read_csv('abalone.csv', sep=',', skipinitialspace=True)
# from sklearn.preprocessing import StandardScaler

features = ['length', 'diameter', 'height', 'whole weight', 'shucked weight', 'viscera weight', 'shell weight']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:, ['rings']].values
# Standardizing the features# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more
# options can be specified also print(abalone_dataset.corr()) x = StandardScaler().fit_transform(x)


pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents
                           , columns=['principal component 1', 'principal component 2', 'principal component 3'])

finalDf = pd.concat([principalDf, df[['rings']]], axis=1)
print(principalDf)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)
targets = [0, 10]
colors = ['r', 'g', 'b']
for target, color in zip(targets, colors):
    if target < 10:
        indicesToKeep = finalDf['rings'] <= 10
    else:
        indicesToKeep = finalDf['rings'] > 10
    print(finalDf['rings'] > 10)
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c=color
               , s=20)
ax.legend(["0 to 10", "10 to 30", "20 to 30"])
ax.grid()
# print(pca.explained_variance_ratio_)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
# print(finalDf.head())
# plt.show()

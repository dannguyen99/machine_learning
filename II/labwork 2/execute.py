import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

x = np.array([1, 2, 9, 12, 20])
print(x)
model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='single')
x = np.reshape(-1, 1)
model.fit(x)
labels = model.labels_
print(labels)



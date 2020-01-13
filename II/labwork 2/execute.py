import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def plot_dendrogram_all(dataset, dataset_name):
    X = dataset.data
    fig_dendrogram, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
    fig_dendrogram.suptitle("Agglomerative Clustering Dendrogram for " + dataset_name + " Dataset\n"
                                                                                        "X axis is the number of "
                                                                                        "points in node (or index of "
                                                                                        "point if no parenthesis).")
    plot_dendrogram(AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='single').fit(X),
                    truncate_mode='level', p=3, ax=ax1)
    ax1.set_title('Single Linkage')
    plot_dendrogram(AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='complete').fit(X),
                    truncate_mode='level', p=3, ax=ax2)
    ax2.set_title('Complete Linkage')
    plot_dendrogram(AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='average').fit(X),
                    truncate_mode='level', p=3, ax=ax3)
    ax3.set_title('Average Linkage')
    plot_dendrogram(AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='ward').fit(X),
                    truncate_mode='level', p=3, ax=ax4)
    ax4.set_title('Ward Linkage')


def plot_dendrogram_single(array):
    fig_dendrogram_single, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig_dendrogram_single.suptitle("Dendrogram for One single array Input")
    dendrogram(linkage(array, method='single'), ax=ax1)
    dendrogram(linkage(array, method='complete'), ax=ax2)


def visualize_ahc_cluster(dataset, dataset_name, linkage_type='ward', feature_set=None):
    if feature_set is None:
        feature_set = [0, 1]
    X = dataset.data
    Y = dataset.target
    no_cluster = len(dataset.target_names)
    feature_names = dataset.feature_names
    first, second = feature_set[0], feature_set[1]
    model = AgglomerativeClustering(n_clusters=no_cluster, affinity='euclidean', linkage=linkage_type)
    model.fit(X)
    labels = model.labels_
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    title = dataset_name + " Dataset with Linkage " + linkage_type
    fig.suptitle(title, fontsize=18)
    colors = ['r', 'g', 'b']
    for i, color in zip(range(no_cluster), colors):
        ax1.scatter(X[Y == i, first], X[Y == i, second], s=50, marker='o', color=color)
    ax1.set_title('Actual')
    for i, color in zip(range(no_cluster), colors):
        ax2.scatter(X[labels == i, first], X[labels == i, second], s=50, marker='o', cmap='jet')
    ax2.set_title('Predicted')
    ax1.set_xlabel(feature_names[first])
    ax1.set_ylabel(feature_names[second])
    ax2.set_xlabel(feature_names[first])
    ax2.set_ylabel(feature_names[second])
    ax1.legend(dataset.target_names)
    return labels


def main(show_plot=False):
    x = np.array([[1], [2], [9], [12], [20]])
    # plot the dendrograms of single array
    # plot_dendrogram_single(x)

    # load Iris dataset
    iris = datasets.load_iris()
    iris_name = "Iris"

    # plot the dendrograms of Iris
    plot_dendrogram_all(dataset=iris, dataset_name=iris_name)

    # apply ahc with Iris, compare with actual data
    visualize_ahc_cluster(dataset=iris, dataset_name=iris_name, linkage_type='complete', feature_set=[0, 3])

    # show the plots
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(show_plot=True)

import numpy as np
from sklearn import datasets, metrics
from matplotlib import pyplot as plt

from sklearn.cluster import AgglomerativeClustering, KMeans
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
    # fig_dendrogram, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
    # fig_dendrogram.suptitle("Agglomerative Clustering Dendrogram for " + dataset_name + " Dataset\n"
    #                                                                                     "X axis is the number of "
    #                                                                                     "points in node (or index of "
    #                                                                                     "point if no parenthesis).")
    plt.figure()
    plot_dendrogram(AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='single').fit(X),
                    truncate_mode='level', p=3)
    plt.title('Single Linkage of ' + dataset_name)
    plt.xlabel("The number of points in node (or index of point if no parenthesis).")
    plt.figure()
    plot_dendrogram(AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='complete').fit(X),
                    truncate_mode='level', p=3)
    plt.title('Complete Linkage of ' + dataset_name)
    plt.xlabel("The number of points in node (or index of point if no parenthesis).")
    plt.figure()
    plot_dendrogram(AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='average').fit(X),
                    truncate_mode='level', p=3)
    plt.title('Average Linkage of ' + dataset_name)
    plt.xlabel("The number of points in node (or index of point if no parenthesis).")
    plt.figure()
    plot_dendrogram(AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='ward').fit(X),
                    truncate_mode='level', p=3)
    plt.title('Ward Linkage of ' + dataset_name)
    plt.xlabel("The number of points in node (or index of point if no parenthesis).")


def plot_dendrogram_single(array):
    fig_dendrogram_single, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig_dendrogram_single.suptitle("Dendrogram for One single array Input")
    ax1.set_title("Single Linkage")
    ax2.set_title('Complete Linkage')
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


def visualize_elbow_method(dataset, dataset_name):
    wcss = []
    X = dataset.data
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='random', n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.figure()
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method for ' + dataset_name)
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')


def visualize_kmeans_cluster(dataset, dataset_name, feature_set=None, use_PCA=False):
    if feature_set is None:
        feature_set = [0, 1]
    X = dataset.data
    Y = dataset.target
    no_cluster = len(dataset.target_names)
    feature_names = dataset.feature_names
    first, second = feature_set[0], feature_set[1]
    model = KMeans(n_clusters=no_cluster, init='random', n_init=10)
    model.fit(X)
    labels = model.labels_
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    title = dataset_name + " Dataset with k = " + str(no_cluster) + " clusters"
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


def clustering_evaluate(dataset, dataset_name, is_print=True):
    results = {}
    X = dataset.data
    no_cluster = len(dataset.target_names)
    model = KMeans(n_clusters=no_cluster, init='random', n_init=10)
    model.fit(X)
    labels_true = dataset.target
    labels_pred = model.labels_
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
        print("Evaluating", dataset_name)
        print("Accuracy =", ACC)
        print("Adjusted Rand index =", ARI)
        print("Mutual Information =", MI)
        print("Normalized Mutual Information =", NMI)
        print("Jaccard Score = ", JC)
        print()
    return results


def main(show_plot=False):
    # 1.1 & 1.2
    # plot the dendrograms of single array
    x = np.array([[1], [2], [9], [12], [20]])
    plot_dendrogram_single(x)

    # 1.3
    # load Iris dataset
    iris = datasets.load_iris()
    iris_name = "Iris"
    # load Wine Dataset
    wine = datasets.load_wine()
    wine_name = "Wine"
    # plot the dendrograms of Iris
    plot_dendrogram_all(dataset=iris, dataset_name=iris_name)
    # plot the dendrograms of Wine
    plot_dendrogram_all(dataset=wine, dataset_name=wine_name)
    # apply ahc with Iris, compare with actual data
    visualize_ahc_cluster(dataset=iris, dataset_name=iris_name, linkage_type='complete', feature_set=[0, 1])
    # apply ahc with Wine, compare with actual data
    visualize_ahc_cluster(dataset=wine, dataset_name=wine_name, linkage_type='single', feature_set=[0, 1])

    # 2.1 & 2.2
    # apply kmeans with Iris, compare with actual data
    visualize_kmeans_cluster(dataset=iris, dataset_name=iris_name, feature_set=[0, 1])
    # apply kmeans with Wine, compare with actual data
    visualize_kmeans_cluster(dataset=wine, dataset_name=wine_name, feature_set=[0, 1])

    # 2.3
    # using elbow method to determine best number of cluster
    visualize_elbow_method(iris, iris_name)
    # using elbow method to determine best number of cluster
    visualize_elbow_method(wine, wine_name)

    # 2.4
    # Evaluate Iris dataset
    clustering_evaluate(dataset=iris, dataset_name=iris_name, is_print=True)
    # Evaluate Wine dataset
    clustering_evaluate(dataset=wine, dataset_name=wine_name, is_print=True)

    # show the plots
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(show_plot=True)

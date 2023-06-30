# 10
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from scipy.spatial import distance
from scipy.sparse.csgraph import laplacian
from scipy import linalg
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import spectral_embedding


def getKMeansSteps(X, k, centroids):
    y_pred = []
    intermediate_centers = []
    k_means = KMeans(n_clusters=k, max_iter=1, init=centroids, n_init=1)
    c_hat = centroids
    for i in range(100):
        intermediate_centers.append(c_hat)
        y_hat = k_means.fit_predict(X)
        c_hat = k_means.cluster_centers_
        y_pred.append(y_hat)
        k_means = KMeans(n_clusters=k, max_iter=1, init=c_hat, n_init=1)

    return y_pred, intermediate_centers


def plot_kmean_sequence(X, k, centroids):
    y_pred_1, centers_1 = getKMeansSteps(X, k, centroids)

    steps = [0, 1, 2, 3, 4, 10, 99]

    fig, ax = plt.subplots(1, 7, figsize=(10, 10))

    ind = 0
    for i in steps:
        ax[0, ind].scatter(X[:, 0], X[:, 1], s=50, c=y_pred_1[i])
        ax[0, ind].plot(
            centers_1[i].transpose()[0],
            centers_1[i].transpose()[1],
            marker="*",
            color="red",
            ls="none",
            ms=10,
        )
        ax[0, ind].set_title("Step = " + str(i))
        ind = ind + 1
    plt.show()


def kmeans_scipy(X_blobs, k):
    """scikit-learn implementation of k-means by default uses random
    ten random re-starts (and then chooses the cluster solution
    resulting in the lowest distortion).
    """
    kmeans = KMeans(n_clusters=k, random_state=111)
    y_blobs_pred = kmeans.fit_predict(X_blobs)
    plt.scatter(X_blobs[:, 0], X_blobs[:, 1], s=80, c=y_blobs_pred)
    plt.show()


def plot_distortion(n_clusters_list, X):
    """
    Plot the distortion (in-cluster variance) on the y-axis and
    the number of clusters in the x-axis

    :param n_clusters_list: List of number of clusters to explore
    :param X: np array of data points
    """
    distortion_list = []
    for k in n_clusters_list:
        kmeans = KMeans(n_clusters=k, random_state=111).fit(X)
        distortion = kmeans.inertia_
        distortion_list.append(distortion)

    plt.plot(n_clusters_list, distortion_list, "o-")
    plt.xlabel("Number of clusters")
    plt.ylabel("Distortion")
    plt.show()


def plot_silhouette(n_clusters_list, X):
    """
    Plot the silhouette score on the y-axis and
    the number of clusters in the x-axis
    :param n_clusters_list: List of number of clusters to explore
    :param X: np array of data points
    """
    silhouette_list = []
    for k in n_clusters_list:
        kmeans = KMeans(n_clusters=k, random_state=111)
        y_pred = kmeans.fit_predict(X)
        silhouette = silhouette_score(X, y_pred)
        silhouette_list.append(silhouette)

    plt.plot(n_clusters_list, silhouette_list, "o-")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette width")
    plt.show()


def compute_bic(kmeans, X, clustering_method="kmeans"):
    """
    Computes the BIC metric

    Usage:
    kmeans = KMeans(n_clusters=4, random_state=111).fit(X_blobs)
    bic = compute_bic(kmeans, X_blobs)

    :param kmeans: clustering object from scikit learn
    :param X: np array of data points
    :return: BIC
    """
    # Adapted from: https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans

    # number of clusters
    k = kmeans.n_clusters
    labels = kmeans.labels_

    if clustering_method == "spectral":
        centers = [np.array([np.mean(X[labels == i], axis=0) for i in range(k)])]
    else:
        centers = [kmeans.cluster_centers_]

    # size of the clusters
    n = np.bincount(labels)
    # size of data set
    N, D = X.shape

    # compute variance for all clusters beforehand
    cl_var = (1.0 / (N - k) / D) * sum(
        [
            sum(
                distance.cdist(X[np.where(labels == i)], [centers[0][i]], "euclidean")
                ** 2
            )
            for i in range(k)
        ]
    )

    LL = np.sum(
        [
            n[i] * np.log(n[i])
            - n[i] * np.log(N)
            - ((n[i] * D) / 2) * np.log(2 * np.pi * cl_var)
            - ((D / 2) * (n[i] - 1))
            for i in range(k)
        ]
    )

    d = (k - 1) + 1 + k * D
    const_term = (d / 2) * np.log(N)

    BIC = LL - const_term

    return BIC


def plot_bic(n_clusters_list, X):
    """
    Plot the BIC on the y-axis and the number of clusters in the x-axis
    :param n_clusters_list: List of number of clusters to explore
    :param X: np array of data points
    """
    bic_list = []
    for k in n_clusters_list:
        kmeans = KMeans(n_clusters=k, random_state=111).fit(X)
        bic = compute_bic(kmeans, X)
        bic_list.append(bic)

    plt.plot(n_clusters_list, bic_list, "o-")
    plt.xlabel("Number of clusters")
    plt.ylabel("BIC")
    plt.show()


def get_gaussian_kernel_similarity(X, gamma=1):
    """
    Computes the similarity matrix
    :param X: np array of data
    :param gamma: the width of the kernel
    :return: similarity matrix
    """

    similarity = pairwise_kernels(X, metric="rbf", gamma=gamma)

    return similarity


def get_adjacency(S, connectivity="full"):
    """
    Computes the adjacency matrix
    :param S: np array of similarity matrix
    :param connectivity: type of connectivity
    :return: adjacency matrix
    """

    if connectivity == "full":
        adjacency = S
    elif connectivity == "epsilon":
        epsilon = 0.5
        adjacency = np.where(S > epsilon, 1, 0)
    else:
        raise RuntimeError("Method not supported")

    return adjacency


def get_adjacency_knn(S, k):
    # S: similarity matrix
    # k: number of neighbors

    S = np.array(S)
    # k+1 because include_self. -S to pass from similarity to distance, +translation to avoid negative values
    G = kneighbors_graph(
        -S + S.max(),
        k + 1,
        metric="precomputed",
        mode="connectivity",
        include_self=True,
    ).toarray()
    W = (G + G.T).astype(bool) * S

    return W


def get_adjacency_knn_OP(S, k):
    # S: similarity matrix
    # k: number of neighbors
    S = np.array(S)
    neighbours = []

    for item in S:
        # plus 1 because list includes self
        idx = np.argsort(item)[-(k + 1) :]  # noqa: E203
        # in idx, 0 represents smallest value
        # take all indexes greater than n_items - k
        # neighbours.append([1 if i >= (len(item) - (k+1)) else 0 for i in idx])
        neighbours.append([1 if i in idx else 0 for i in range(len(item))])

    W = np.zeros(S.shape)
    for i in range(len(W)):
        for j in range(len(W)):
            W[i][j] = S[i][j] if neighbours[i][j] or neighbours[j][i] else 0

    return W


def spectral_clustering(W, n_clusters, random_state=111):
    """
    Spectral clustering
    :param W: np array of adjacency matrix
    :param n_clusters: number of clusters
    :return: tuple (kmeans, proj_X, eigenvals_sorted)
        WHERE
        kmeans scikit learn clustering object
        proj_X is np array of transformed data points
        eigenvals_sorted is np array with ordered eigenvalues

    """
    # Compute eigengap heuristic
    L = laplacian(W, normed=True)
    eigenvals, _ = linalg.eig(L)
    eigenvals = np.real(eigenvals)
    eigenvals_sorted = eigenvals[np.argsort(eigenvals)]

    # Create embedding
    random_state = np.random.RandomState(random_state)
    proj_X = spectral_embedding(
        W, n_components=n_clusters, random_state=random_state, drop_first=False
    )

    # Cluster the points using k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(proj_X)

    return kmeans, proj_X, eigenvals_sorted


def get_spectral_clustering_labels(X):
    n_clusters = 4
    random_state = 0
    data = X

    gamma = 1
    S = get_gaussian_kernel_similarity(data, gamma)
    W = get_adjacency(S)
    kmeans, proj_X, eigenvals_sorted = spectral_clustering(
        W, n_clusters, random_state=random_state
    )

    plt.scatter(X[:, 0], X[:, 1], s=50, c=kmeans.labels_)
    plt.show()

    return kmeans.labels_


def spectral_clustering_sklearn(data, n_clusters, random_state=111):
    labels_sc = SpectralClustering(
        n_clusters=n_clusters, random_state=random_state
    ).fit_predict(data)


def plot_metrics(n_clusters_list, metric_dictionary):
    """
    Plots metric dictionary (auxilary function)
    [Optional]

    :param n_clusters_list: List of number of clusters to explore
    :param metric_dictionary:
    """
    fig = plt.figure(figsize=(12, 10), dpi=80)
    i = 1

    for metric in metric_dictionary.keys():
        plt.subplot(3, 2, i)

        if metric == "Eigengap":
            clusters = len(n_clusters_list)
            eigenvals_sorted = metric_dictionary[metric]
            plt.scatter(
                range(1, len(eigenvals_sorted[: clusters * 2]) + 1),
                eigenvals_sorted[: clusters * 2],
            )
            plt.xlabel("Eigenvalues")
            plt.xticks(range(1, len(eigenvals_sorted[: clusters * 2]) + 1))
        else:
            plt.plot(n_clusters_list, metric_dictionary[metric], "-o")
            plt.xlabel("Number of clusters")
            plt.xticks(n_clusters_list)
        plt.ylabel(metric)
        i += 1


def get_heuristics_spectral(W, n_clusters_list, plot=True):
    """
    Calculates heuristics for optimal number of clusters with Spectral Clustering

    :param W: np array of adjacency matrix
    :param n_clusters_list: List of number of clusters to explore
    :plot: bool, plot the metrics if true
    """
    silhouette_list = []
    eigengap_list = []

    df_labels = pd.DataFrame()

    for k in n_clusters_list:
        kmeans, proj_X, eigenvals_sorted = spectral_clustering(W, k)
        y_pred = kmeans.labels_
        df_labels[str(k)] = y_pred

        if k == 1:
            silhouette = np.nan
        else:
            silhouette = silhouette_score(proj_X, y_pred)
        silhouette_list.append(silhouette)

    metric_dictionary = {
        "Silhouette": silhouette_list,
        "Eigengap": eigenvals_sorted,
    }

    if plot:
        plot_metrics(n_clusters_list, metric_dictionary)
        return df_labels
    else:
        return df_labels, metric_dictionary


def compare_k_values(X, n_clusters_list):
    """
    Compares k values for Spectral Clustering

    :param X: np array of data points
    :param n_clusters_list: list of k values to compare
    """
    S = get_gaussian_kernel_similarity(X, gamma=1)
    W = get_adjacency(S)
    get_heuristics_spectral(W, n_clusters_list)


def clustering_example(df):
    data = StandardScaler().fit_transform(df[["ch_time_sum", "ch_total_clicks"]])
    time = data[:, [0]]
    clicks = data[:, [1]]

    # sum similarity matrices for multiple columns
    S1 = pairwise_kernels(time, metric="rbf", gamma=5)
    S2 = pairwise_kernels(clicks, metric="rbf", gamma=5)
    S = (S1 + S2) / 2

    W = get_adjacency(S)

    get_heuristics_spectral(W, list(range(2, 10)))


def visualize_clusters(df, f1_name, f2_name, n_clusters_list):
    data = StandardScaler().fit_transform(df[[f1_name, f2_name]])
    f1 = data[:, [0]]
    f2 = data[:, [1]]

    S1 = pairwise_kernels(f1, metric="rbf", gamma=1)
    S2 = pairwise_kernels(f2, metric="rbf", gamma=1)
    S = (S1 + S2) / 2

    get_heuristics_spectral(S, n_clusters_list)

# Important imports
import numpy as np
import matplotlib.pyplot as plt
from tslearn.metrics import cdist_dtw

from scipy.spatial import distance

from sklearn.metrics.pairwise import pairwise_kernels
from clustering import get_adjacency, get_heuristics_spectral


def clustering_fixed_time_intervals(X):
    # Data must have same length of time-series
    norms = np.linalg.norm(X, axis=1)
    data_normalized = X / norms[:, np.newaxis]
    S = pairwise_kernels(data_normalized, metric="rbf", gamma=1)
    W = get_adjacency(S)
    n_cluster_list = range(2, 10)
    df_labels = get_heuristics_spectral(W, n_cluster_list)


def get_distance_matrix_dtw_fixed_window(X, metric="euclidean", window=2):
    """
    calculates distance matrix given a metric
    :param X: np.array with students' time-series
    :param metric: str distance metric to compute
    :param window: int for DTW
    :return: np.array with distance matrix
    """
    norms = np.linalg.norm(X, axis=1)
    data_normalized = X / norms[:, np.newaxis]

    # Dynamic time warping allows us to align to sequences in an
    #  optimal way by choosing a window size w larger than 0.
    if metric == "dtw":
        distance_matrix = cdist_dtw(
            data_normalized, global_constraint="sakoe_chiba", sakoe_chiba_radius=window
        )
    else:
        distance_vector = distance.pdist(data_normalized, metric)
        distance_matrix = distance.squareform(distance_vector)
    return distance_matrix


def get_affinity_matrix(D, gamma=1):
    """
    computes the similarity matrix for us based on the pairwise distances.
    calculates affinity matrix from distance matrix
    :param D: np.array distance matrix
    :param gamma: float coefficient for Gaussian Kernel
    :return:
    """
    S = np.exp(-gamma * D**2)
    return S


def clustering_dynamic_time_warping(data):
    D = get_distance_matrix_dtw_fixed_window(data, metric="dtw", window=6)
    S = get_affinity_matrix(D)
    W = get_adjacency(S)
    n_cluster_list = range(2, 10)
    df_labels = get_heuristics_spectral(W, n_cluster_list)


def view_clusters(data, labels, ylim=70, xlabel="Biweeks"):
    """
    visualize the different time-series of students belonging to each cluster.
    :param data: np.array with students' time-series
    :param labels: np.array predicted labels from clustering model
    :return:
    """
    _, biweeks = data.shape
    clusters = np.unique(labels).shape[0]
    fig, axs = plt.subplots(1, clusters, figsize=(16, 4), facecolor="w", edgecolor="k")
    axs = axs.ravel()

    for i in range(clusters):
        students_cluster = data[labels == i]
        number_students = students_cluster.shape[0]
        for student in range(number_students):
            axs[i].bar(range(biweeks), students_cluster[student], alpha=0.3)

        axs[i].set_ylim([0, ylim])
        axs[i].set_title("Group {0}".format(i))
        axs[i].set_ylabel("Hours using platform")
        axs[i].set_xlabel(xlabel)


def plot_students_group(data, labels):
    """
    Plot the students time-series
    :param data: np.array with students' time-series
    :param labels: pd.Series indicating the labels of the students
    :return:
    """
    for group in np.unique(labels):
        subdata = data[labels == group]
        subindex = labels[labels == group].index
        students, biweeks = subdata.shape

        rows = int(np.ceil(students / 6))
        fig, axs = plt.subplots(
            rows,
            6,
            figsize=(16, rows * 3),
            sharex=True,
            sharey=True,
            facecolor="w",
            edgecolor="k",
        )

        axs = axs.ravel()
        for i in range(students):
            axs[i].bar(range(biweeks), subdata[i], alpha=0.8)
            axs[i].set_ylim([0, 50])
            axs[i].set_title("Student {0}".format(subindex[i]))

        fig.suptitle("GROUP {}".format(group))
        fig.supxlabel("Biweek")
        fig.supylabel("Usage of platform (hours)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    X = [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2]]
    print(get_distance_matrix_dtw_fixed_window(X, metric="dtw", window=10))
    print(get_distance_matrix_dtw_fixed_window(X, metric="euclidean", window=10))

    X2 = [[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 2, 0]]
    print(get_distance_matrix_dtw_fixed_window(X2, metric="dtw", window=10))
    print(get_distance_matrix_dtw_fixed_window(X2, metric="euclidean", window=10))

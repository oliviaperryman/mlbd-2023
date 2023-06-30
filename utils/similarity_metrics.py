import numpy as np
from Levenshtein import distance
from difflib import SequenceMatcher

from scipy.spatial.distance import pdist, squareform

from sklearn.kernel_approximation import pairwise_kernels


def euclidean_distance(l1, l2):
    return np.linalg.norm(l1 - l2)


def jaccard_difference(l1, l2):
    l1 = set(l1)
    l2 = set(l2)

    return 1 - (len(l1.intersection(l2)) / len(l1.union(l2)))


def hammington_distance(l1, l2):
    # Strings of equal length
    return np.sum(l1 != l2)


def levenstein_distance(l1, l2):
    """minimal number of single character edits (insertion, deletion, substitution) to change one string into the other"""
    return distance(l1, l2)


def longest_common_subsequence(l1, l2):
    return SequenceMatcher(None, l1, l2).find_longest_match(0, len(l1), 0, len(l2))


def kullback_leibler_divergence(p, q):
    # measures difference between two probability distributions (relative entropy)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def apply_difference(X, metric_fn):
    S = []
    for s1 in X:
        entry = []
        for s2 in X:
            entry.append(metric_fn(s1, s2))
        S.append(entry)

    return np.array(S)


def get_gaussian_kernel_similarity(X, gamma=1):
    """
    Computes the similarity matrix
    :param X: np array of data
    :param gamma: the width of the kernel
    :return: similarity matrix
    """

    similarity = pairwise_kernels(X, metric="rbf", gamma=gamma)

    return similarity


def cosine_similarity(X):
    similarity = pairwise_kernels(X, metric="cosine")

    return similarity


def generic_similarity(X, metric):
    # metric_options are [‘additive_chi2’, ‘chi2’, ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’, ‘laplacian’, ‘sigmoid’, ‘cosine’]
    similarity = pairwise_kernels(X, metric=metric)

    return similarity


def jaccard_similarity_pairwise(X):
    S = squareform(
        pdist(
            X,
            metric=lambda x, y: float(
                len(x[0].intersection(y[0])) / len(x[0].union(y[0]))
            ),
        )
    )


def frobenius_norm(X):
    return np.linalg.norm(X, ord="fro")


def jensen_shannon_divergence(p, q):
    # measures similarity between two probability distributions
    m = 0.5 * (p + q)
    return 0.5 * (kullback_leibler_divergence(p, m) + kullback_leibler_divergence(q, m))


def hellinger_distance(p, q):
    # measures similarity between two probability distributions
    return 1 / np.sqrt(2) * np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

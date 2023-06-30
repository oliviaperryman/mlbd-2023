from scipy import stats
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score


def pearson_corr(x, y):
    """Pearson correlation coefficient between two variables.
    Captures the linear relationship only.
    If two variables are independent, the correlation coefficient is 0.
    If correlation is 0, the variables are not necessarily independent
     (could be a different dependence other than linear)
    """

    # r, p = stats.pearsonr(x, y)

    # covariance = np.cov(x, y)
    covariance = np.sum((x - np.mean(x)) * (y - np.mean(y))) / (len(x) - 1)

    corr = covariance / (np.std(x) * np.std(y))

    return corr


def mutual_information(x, y):
    """Mutual information between two variables.
    Defines the amount of information obtained about one variable through observing the other.
    If two variables are independent, the mutual information is 0 because p(xy) = p(x)p(y)
    and log(1) = 0
    If continuous variables, discretize them first. Sample and bin them. However, different binning
    can lead to different results.
    """
    return adjusted_mutual_info_score(x, y)

    # Manually, not working
    # pk = np.histogram(x, bins="auto")[0] / len(x)
    # qk = np.histogram(y, bins="auto")[0] / len(y)
    # KL_divergence = np.sum(pk * np.log(pk / qk))
    # return KL_divergence


if __name__ == "__main__":
    x = np.random.normal(2, 1, 100)
    y = np.random.normal(1, 1, 100)

    print(pearson_corr(x, y))
    print(mutual_information(x, y))

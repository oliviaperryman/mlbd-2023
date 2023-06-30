import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
)


def AIC(log_liklihood, num_samples, num_params):
    """
    Lower AIC scores are better, and AIC penalizes models that use more parameters. So if two models explain the same amount of variation, the one with fewer parameters will have a lower AIC score and will be the better-fit model. The larger the sample size, the more params we can use.

    :param log_liklihood: Log liklihood of the model
    :param num_samples: Number of samples
    :param num_params: Number of parameters

    """
    return (2 * num_params / num_samples) - (2 * log_liklihood / num_samples)


def BIC(log_liklihood, num_samples, num_params):
    """
    Higher punishment for complex models than AIC. Lower BIC value indicates lower penalty terms hence a better model.
    """
    return (num_params * np.log(num_samples)) - (2 * log_liklihood)


def compute_scores(clf, X_train, y_train, X_test, y_test, roundnum=3, report=False):
    """
    Train clf (binary classification) model on X_train and y_train, predict on X_test. Evaluate predictions against ground truth y_test.
    Inputs: clf, training set (X_train, y_train), test set (X_test, y_test)
    Inputs (optional): roundnum (number of digits for rounding metrics), report (print scores)
    Outputs: accuracy, AUC
    """
    # Fit the clf predictor (passed in as an argument)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate roc AUC score
    AUC = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    # Print classification results
    if report:
        print(classification_report(y_test, y_pred))

    return round(accuracy, roundnum), round(AUC, roundnum)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def balanced_accuracy(y_true, y_pred):
    """easy to interpret: average accuracy over all classes
    takes into account class imbalance
    works for multiclass
    """
    return balanced_accuracy_score(y_true, y_pred)


def specificity(y_true, y_pred):
    """True Negative Rate: proportion of negative cases that were correctly identified"""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp)


def sensitivity(y_true, y_pred):
    """True Positive Rate: proportion of positive cases that were correctly identified"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn)


def auc_definition(y_true, y_pred):
    """Area under the ROC curve"""
    return roc_auc_score(y_true, y_pred)


def mae(true_vals, pred_vals):
    return np.mean(np.abs(true_vals - pred_vals))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


if __name__ == "__main__":
    # Question 8
    # M1: num_movies ~ x1
    # M2: num_movies ~ 1 + x1
    # M3: num_movies ~ 1 + x1 + x2

    # Let the log-likelihood of the models be as follows: M1: 128, M2: 130.7, M3: 131. Assume you have 100 observations.
    n_samples = 100
    print(AIC(128, n_samples, 1))
    print(AIC(130.7, n_samples, 1))
    print(AIC(131, n_samples, 2))

    # # Question 9
    print(BIC(128, n_samples, 1))
    print(BIC(130.7, n_samples, 1))
    print(BIC(131, n_samples, 2))

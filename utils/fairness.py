import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


def true_positive_rate(df):
    """Calculate equal opportunity (true positive rate)."""

    # Confusion Matrix
    cm = confusion_matrix(df["y"], df["y_pred"])
    TN, FP, FN, TP = cm.ravel()

    # Total population
    N = TP + FP + FN + TN

    # True positive rate
    TPR = TP / (TP + FN)

    return TPR


def compute_equal_opportunity(df, threshold):
    """Under equal opportunity we consider a model to be fair if the
    TPRs of the privileged and unprivileged groups are equal. In
    practice, we will give some leeway for statistic uncertainty.
    We can require the differences to be less than a certain cutoff
    (Equation 2). For our analysis, we have taken the ratio. In this
    case, we require the ratio to be larger than some cutoff
    (Equation 3). This ensures that the TPR for the unprivileged group
    is not significantly smaller than for the privileged group.Under
    equal opportunity we consider a model to be fair if the TPRs of the
    privileged and unprivileged groups are equal. In practice, we will
    give some leeway for statistic uncertainty. We can require the
    differences to be less than a certain cutoff (Equation 2). For
    our analysis, we have taken the ratio. In this case, we require
    the ratio to be larger than some cutoff (Equation 3). This ensures
    that the TPR for the unprivileged group is not significantly smaller
    than for the privileged group."""
    TPR_1 = true_positive_rate(df[df["gender"] == "M"])
    TPR_2 = true_positive_rate(df[df["gender"] == "F"])

    return TPR_1 / TPR_2 > threshold


# For equal opportunity, we directly compare the difference between TPRs of the sensitive attributes.
# We define our significance cutoff at 0.1, stating any difference below 10% can be attributed to random chance.
def stats_eq_opp(df, attr, stat="TPR", cutoff=0.1, indexs=[0, 1]):
    """
    df: dataframe with sensitive attribute and TPRs
    """
    TPR_0, TPR_1 = df[stat][indexs[0]], df[stat][indexs[1]]
    equal_opp = np.abs(np.round(TPR_1 - TPR_0, 3))
    equal_opp_ratio = np.round(np.minimum(TPR_0, TPR_1) / np.maximum(TPR_0, TPR_1), 3)

    print("Sensitive Attr:", attr, "\n")

    print("------------------------------------")
    print("|Equal Opportunity| < Cutoff?", np.abs(equal_opp) > cutoff)
    print("------------------------------------")
    print("TPR0 (", df[attr][indexs[0]], ") =", TPR_0)
    print("TPR1 (", df[attr][indexs[1]], ") =", TPR_1)
    print("Equal Opportunity:", equal_opp)
    print("Cutoff:", cutoff)

    print("\n------------------------------------")
    print("Equal Opportunity Ratio?", equal_opp_ratio)
    print("------------------------------------")


def get_heatmap(df, attr, func, stat="TPR", cutoff=0.1):
    """
    plot heatmap of TPRs
    """
    size = df[attr].size
    data = df[stat]
    heatmap = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            heatmap[i, j] = func([data[i], data[j]])

    return heatmap


def plot_heatmaps(combined_df):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    heatmap = get_heatmap(
        combined_df, "gender_country", lambda x: np.round(np.abs(x[0] - x[1]), 3)
    )
    sns.heatmap(
        heatmap,
        ax=ax[0],
        xticklabels=combined_df["gender_country"],
        yticklabels=combined_df["gender_country"],
        annot=True,
        vmin=0,
        vmax=1,
    )
    ax[0].set_title("|Equal Opportunity| < cutoff = 0.1 ?")

    heatmap = get_heatmap(
        combined_df,
        "gender_country",
        lambda x: np.round(np.minimum(x[0], x[1]) / np.maximum(x[0], x[1]), 3),
    )
    sns.heatmap(
        heatmap,
        ax=ax[1],
        xticklabels=combined_df["gender_country"],
        yticklabels=combined_df["gender_country"],
        annot=True,
        vmin=0,
        vmax=1,
    )
    ax[1].set_title("Equal Opportunity Ratio > cutoff = 0.1 ?")

    plt.show()


def false_negative_rate(df):
    """Calculate false negative rate"""

    # Confusion Matrix
    cm = confusion_matrix(df["y"], df["y_pred"])
    TN, FP, FN, TP = cm.ravel()

    # False negative rate
    FNR = FN / (TP + FN)

    return FNR


# For demographic parity, we compare the difference between the PPPs
#  of the sensitive attributes.
def compute_demographic_parity(df):
    """Calculate PPP for subgroup of population
    **Demographic Parity** states that the proportion of each segment
     of a protected class (e.g. gender) should receive the positive
     outcome at equal rates. In other words, the probability of a
     positive outcome (denoted as PPP) should be the same independent
     of the value of the protected attribute.
    """

    # Confusion Matrix
    cm = confusion_matrix(df["y"], df["y_pred"])
    TN, FP, FN, TP = cm.ravel()

    # Total population
    N = TP + FP + FN + TN

    # predicted as positive
    PPP = (TP + FP) / N

    return PPP


def demographic_parity(df):
    ppp_m = compute_demographic_parity(df[df["gender"] == "M"])
    ppp_f = compute_demographic_parity(df[df["gender"] == "F"])


def compute_equalized_odds(df):
    """Calculate FPR and TPR for subgroup of population
    requires that the true positive rates (TPR) as well as the
    false positive rates (FPR) are equal across values of the
    sensitive attribute. That is, a similar percentage of the
    groups should both rightfully and wrongfully benefit.

    An advantage of equalized odds is that it does not matter
    how we define our target variable. Suppose instead we had
    Y = 0 leads to a benefit. In this case the interpretations
    of TPR and FPR swap. TPR now captures the wrongful benefit
    and FPR now captures the rightful benefit. Equalized odds
    already uses both of these rates so the interpretation remains
    the same. In comparison, the interpretation of equal opportunity
    changes as it only considers TPR.
    """

    # Confusion Matrix
    cm = confusion_matrix(df["y"], df["y_pred"])
    TN, FP, FN, TP = cm.ravel()

    # True positive rate
    TPR = TP / (TP + FN)

    # False positive rate
    FPR = FP / (FP + TN)

    return [TPR, FPR]


def equalized_odds(df):
    tpr_m, fpr_m = compute_equalized_odds(df[df["gender"] == "M"])
    tpr_f, fpr_f = compute_equalized_odds(df[df["gender"] == "F"])


def compute_predictive_value_parity(df):
    """Calculate predictive value parity scores
    Predictive value-parity equalizes the probability of a
    positive outcome, given a positive prediction (PPV) and
    the probability of a negative outcome given a negative
    prediction (NPV).
    """

    # Confusion Matrix
    cm = confusion_matrix(df["y"], df["y_pred"])
    TN, FP, FN, TP = cm.ravel()

    # Positive Predictive Value
    PPV = TP / (FP + TP)

    # Negative Predictive Value
    NPV = TN / (FN + TN)

    return [PPV, NPV]


def predictive_value_parity(df):
    ppv_m, npv_m = compute_predictive_value_parity(df[df["gender"] == "M"])
    ppv_f, npv_f = compute_predictive_value_parity(df[df["gender"] == "F"])


def convert_prediction_to_binary(df, threshold=0.5):
    y_pred = [1 if grade > threshold else 0 for grade in df["grade"]]


def prevalence(df, group, y_col):
    """the proportion of positive cases to overall cases."""

    # prevalence by category
    prev_group = df.groupby(group)[y_col].mean()

    return prev_group


def accuracy(df):
    """Calculate accuracy through the confusion matrix.
    accuracy is the percentage of correct predictions

    """

    # Confusion Matrix
    cm = confusion_matrix(df["y"], df["y_pred"])
    TN, FP, FN, TP = cm.ravel()

    # Total population
    N = TP + FP + FN + TN

    # Accuracy
    ACC = (TP + TN) / N

    return ACC


def disparate_impact(df):
    """Calculate PPP for subgroup of population"""

    # Confusion Matrix
    cm = confusion_matrix(df["y"], df["y_pred"])
    TN, FP, FN, TP = cm.ravel()

    # Total population
    N = TP + FP + FN + TN

    # predicted as positive
    PPP = (TP + FP) / N

    return PPP

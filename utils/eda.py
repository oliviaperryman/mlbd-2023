# 02
from scipy import stats
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, chi2_contingency
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mutual_info_score

pd.set_option("display.precision", 2)


def descriptive_stats(df, feature_list):
    data = {
        "Mean": np.mean(feature_list, 1),
        "Median": np.median(feature_list, 1),
        "Mode": (stats.mode(feature_list, axis=1, keepdims=True)[0])[:, 0],
        "Variance": np.var(feature_list, 1),
        "Std": np.std(feature_list, 1),
        "Minimum": np.min(feature_list, 1),
        "25%": np.percentile(feature_list, 25, axis=1),
        "75%": np.percentile(feature_list, 75, axis=1),
        "Maximum": np.max(feature_list, 1),
    }
    desc_stats_df = pd.DataFrame(
        data,
        index=df.columns,
    )

    return desc_stats_df


def plot_features(df, title):
    continuous_cols = list(df._get_numeric_data().columns)
    categorical_cols = list(set(df.columns) - set(continuous_cols))
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i, col in enumerate(df.columns):
        ax = axes[i // 3, i % 3]
        data = df[~df[col].isna()]
        if col in continuous_cols:
            sns.histplot(
                data=data[col], bins=50, ax=ax
            )  # Filter out nan values in the features
        elif col in categorical_cols:
            sns.countplot(data=data, x=col, ax=ax)
        else:
            print(col)
    fig.suptitle(title)
    fig.tight_layout()


def plot_correlation(df):
    """
    Builds upper triangular heatmap with pearson correlation between numerical variables

    Instructions
    ------------
    The plot must have:
    - An appropiate title
    - Only upper triangular elements
    - Annotated values of correlation coefficients rounded to three significant
    figures
    - Negative correlation must be blue and possitive correlation red.

    Parameters
    ----------
    df : DataFrame with data


    """
    corr = np.round(df.corr(method="pearson"), 3)
    mask = np.tril(corr)
    ax = plt.axes()
    heatmap = sns.heatmap(corr, annot=True, mask=mask, cmap="RdBu_r")
    ax.set_title("Correlation between variables")
    plt.show()


def univariate(df, feature):
    # Bar count plot
    ax = sns.countplot(data=df, x=feature)
    plt.show()

    # Pie chart
    val_counts = df[feature].value_counts() / np.sum(df[feature].value_counts())
    labels = val_counts.index.to_list()
    plt.pie(val_counts, labels=labels, autopct="%1.1f%%")
    plt.show()

    # Histogram, distribution
    sns.histplot(data=df, x=feature, kde=True)
    plt.show()


def multivariate(df, x, y, categorical=None):
    # Scatter
    sns.scatterplot(data=df, y=y, x=x, hue=categorical)
    plt.show()

    # correlations
    r, p = stats.pearsonr(df[x], df[y])
    print("Pearson correlations:", r, p)

    # scatter between all numerical variables
    sns.pairplot(df)
    plt.show()

    # heatmaps of correlations
    sns.set(font_scale=1.15)
    plt.figure(figsize=(8, 4))
    sns.heatmap(df.corr(), cmap="RdBu_r", annot=True, vmin=-1, vmax=1)
    plt.title("Pearson correlation heatmap")

    sns.set(font_scale=1.15)
    plt.figure(figsize=(8, 4))
    sns.heatmap(
        df.corr(method=mutual_info_score), cmap="RdBu_r", annot=True, vmin=1, vmax=6
    )
    plt.title("Mutual Information heatmap")

    # Regression
    sns.regplot(data=df, y=y, x=x)

    # Plot three features
    sns.jointplot(data=df, y=y, x=x, hue=categorical)
    plt.show()


def mutual_information_discrete(df, categorical_x, categorical_y):
    counts_xy = df[[categorical_x, categorical_y]].value_counts()
    d_xy = pd.DataFrame((counts_xy / (counts_xy.sum()))).reset_index()
    d_xy.columns = [categorical_x, categorical_y, "pxy"]

    counts_x = df[categorical_x].value_counts()
    dx = pd.DataFrame((counts_x / (counts_x.sum()))).reset_index()
    dx.columns = [categorical_x, "px"]

    counts_y = df[categorical_y].value_counts()
    dy = pd.DataFrame(counts_y / (counts_y.sum())).reset_index()
    dy.columns = [categorical_y, "py"]

    d_mi = d_xy.merge(dx, on=categorical_x, how="left").merge(
        dy, on=categorical_y, how="left"
    )

    mi = 0
    for i, row in d_mi.iterrows():
        if row["pxy"] > 0:
            mi += row["pxy"] * np.log(row["pxy"] / (row["px"] * row["py"]))

    return mi


def mutual_info_scipy(df, categorical_x, categorical_y):
    mutual_info_classif(
        df[categorical_x].values.reshape(-1, 1),
        df[categorical_y].values,
        discrete_features=True,
    )


def mutual_information_continous(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    mi = 0.5 * g / c_xy.sum()
    return mi


def test_normality(data):
    k2, p = stats.normaltest(data)
    alpha = 0.01
    print("p = {:g}".format(p))
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print("The null hypothesis can be rejected")
    else:
        print("The null hypothesis cannot be rejected")


def visualize_skewness(a):
    data = skewnorm.rvs(a, size=1000)
    sns.histplot(data=pd.DataFrame(data), kde=True)
    plt.show()
    test_normality(data)


def plot_by_gender(feature, ylim, xlim, bins):
    f, axarr = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

    sns.histplot(
        data=df, x=feature, hue="gender", ax=axarr[0], bins=bins, binrange=(0, xlim)
    )
    axarr[0].set(title="Both", ylim=(0, ylim), xlim=(0, xlim))

    sns.histplot(
        data=df[df.gender == "F"],
        x=feature,
        color="orange",
        ax=axarr[1],
        bins=bins,
        binrange=(0, xlim),
    )
    axarr[1].set(title="Female", ylim=(0, ylim), xlim=(0, xlim))

    sns.histplot(
        data=df[df.gender == "M"], x=feature, ax=axarr[2], bins=bins, binrange=(0, xlim)
    )
    axarr[2].set(title="Male", ylim=(0, ylim), xlim=(0, xlim))
    plt.show()


if __name__ == "__main__":
    DATA_DIR = "data"
    df = pd.read_csv("{}/aggregated_fc.csv".format(DATA_DIR))

    feature_list = [
        df["grade"],
        df["sessions"],
        df["time_in_problem"],
        df["time_in_video"],
        df["lecture_delay"],
        df["content_anticipation"],
        df["mean_playback_speed"],
        df["relative_video_pause"],
        df["submissions"],
        df["submissions_correct"],
        df["clicks_weekend"],
        df["clicks_weekday"],
    ]

    print(descriptive_stats(df, feature_list))

    univariate(df, "category")
    multivariate(df, "grade", "time_in_problem")

    # normally distributed
    test_normality(df["mean_playback_speed"])
    sns.histplot(data=df, x="mean_playback_speed", kde=True)
    plt.show()

    plot_by_gender("submissions_correct", ylim=50, xlim=90, bins=9)
    plot_by_gender("grade", 40, 6, bins=9)

    # Mutual Info
    categorical_x = "gender"
    categorical_y = "category"
    mi = mutual_information_discrete(df, categorical_x, categorical_y)
    print("Mututal Info", mi)

    df_mi = df[(~df[categorical_x].isna() & ~df[categorical_y].isna())]
    x = LabelEncoder().fit_transform(df_mi[categorical_x]).reshape(-1, 1)
    y = LabelEncoder().fit_transform(df_mi[categorical_y]).ravel()
    mi = mutual_info_classif(x, y, discrete_features=True)
    print("Mututal Info scikit", mi)

    x = np.array(df["time_in_problem"])
    y = np.array(df["grade"]).ravel()
    bins = np.floor(np.sqrt(df.shape[0] / 5))
    mutual_information_continous(x, y, int(bins))

    x = np.array(df["time_in_problem"]).reshape(-1, 1)
    mi = mutual_info_regression(x, y, n_neighbors=1)
    print("Mututal info regression scikit:", mi)

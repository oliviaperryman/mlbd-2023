import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_ts(ts, x, y, category=None):
    sns.lineplot(data=ts, x=x, y=y, errorbar="sd", hue=category)
    plt.show()


if __name__ == "__main__":
    DATA_DIR = "data"
    ts = pd.read_csv("{}/time_series_fc.csv".format(DATA_DIR))
    df = pd.read_csv("{}/aggregated_fc.csv".format(DATA_DIR))

    plot_ts(ts, "week", "sessions")

    ts = ts.merge(df[["user", "gender"]], how="left", on="user")
    plot_ts(ts, "week", "sessions", "gender")

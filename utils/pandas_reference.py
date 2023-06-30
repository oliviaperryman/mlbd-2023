import pandas as pd
import numpy as np


def get_number_of_attempts_from_order_by_user(order_id, user_id, df):
    df = df.sort_values("order_id")
    df["aux"] = 1
    df["prev_attempts"] = df.groupby(["user_id", "skill_name"])["aux"].cumsum() - 1
    # Number of correct and incorrect attempts before current attempt
    df["correct_aux"] = (
        df.sort_values("order_id")
        .groupby(["user_id", "skill_name"])["correct"]
        .cumsum()
    )
    df["before_correct_num"] = (
        df.sort_values("order_id")
        .groupby(["user_id", "skill_name"])["correct_aux"]
        .shift(periods=1, fill_value=0)
    )
    df["before_wrong_num"] = df["prev_attempts"] - df["before_correct_num"]


def get_time_series(df):
    """
    reshapes DataFrame from long to wide and returns an np.array
    :param df: pd.DataFrame with data in long format
    :return: np.array with reshaped data
    """
    df_array = (
        df.sort_values(["student_id", "biweek_of_year"], ascending=True)
        .groupby("student_id")
        .agg({"hours": lambda x: list(x)})
    )

    data = np.asarray(df_array.hours.values.tolist())
    return data


if __name__ == "__main__":
    df = pd.read_csv("example.csv")
    df2 = pd.DataFrame()

    # descriptive statistics
    stats = df.describe(include="all")
    percent_missing = df.isnull().sum() * 100 / len(df)

    # value counts
    df.category.value_counts(dropna=False)

    # drop some rows
    df.drop(df[df.gender != "female"].index)

    # replace values
    df.fillna("NaN")
    df = df.replace([99, "99"], np.nan)
    mapping_time = {
        "1 hr": 60,
        "2hrs": 120,
        "2 hours": 120,
        "30 min": 30,
        "45 min": 45,
        "60 minutes": 60,
        "1.5 hours": 90,
    }
    mapping_group = {"a": "A", "b": "B", "c": "C", "aa": "A", "Bb": "B", "cc": "C"}
    df = df.replace({"minutes": mapping_time, "school_group": mapping_group})
    # convert to numeric
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")

    # remove duplicates
    df = df.groupby(["student_id"]).first()

    # count unique by group
    n_games_per_user = df.groupby("user_id")["game_name"].nunique()

    # count by specific value
    num_help_events = (
        df[df["type"] == "HELP"]["user_id"]
        .value_counts()
        .rename_axis("user_id")
        .to_frame("num_help_events")
    )

    # percentage correct
    percentage_correct = (
        df.groupby("user_id")["correct"]
        .mean()
        .apply(lambda x: x * 100)
        .to_frame("percentage_correct")
    )

    # join and merge
    df.join(df2)
    df.reset_index().merge(
        df2.reset_index(),
        how="left",
        on=["event_id", "user_id"],
        suffixes=("_event", "_subtask"),
    )

    # date features
    df[["Year", "Week", "Day"]] = pd.to_datetime(df["start"]).dt.isocalendar()
    df["year_week"] = (df["Year"] - 2015) * 53 + df["Week"]
    df["week"] = df.groupby("user_id")["year_week"].apply(
        lambda x: x - x.iat[0]
    )  # 0 align weeks

    # time series
    ts = (
        df.sort_values(["student_id", "biweek_of_year"], ascending=True)
        .groupby("student_id")
        .agg({"hours": lambda x: list(x)})
    )

    # rolling mean
    pc = df.groupby("week")["percentage_correct"].mean().to_frame().reset_index()
    pc["rolling"] = pc["percentage_correct"].rolling(8).mean().shift(-4)

    # Get subset of rows based on condition
    subset = []
    data = df[df["skill_name"].isin(subset)]

    # string to list
    df["topics"][0].replace("'", "").replace("[", "").replace("]", "").split(", ")

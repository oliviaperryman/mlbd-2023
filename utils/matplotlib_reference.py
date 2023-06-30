import pandas
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    df = pandas.DataFrame()

    # Horizontal and verical lines on graph to show mean
    plt.axhline(
        df[(df["week"] < 50) & (df["max_number_range_cat"] == 1000)][
            "percentage_correct"
        ].mean(),
        color="red",
    )
    plt.axvline(x=df["column_name"].mean(), color="red")

    # Linear regression plot
    sns.lmplot(x="studying_hours", y="quiz_grade", data=df)

    # Bar plot with 95% confidence interval
    sns.barplot(x="Method", y="Accuracy", data=df, errorbar="ci")

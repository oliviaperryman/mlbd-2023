# 06, 07
# PyBKT package imports
from pyBKT.models import Model
from sklearn.metrics import mean_squared_error, roc_auc_score
from preprocessing import create_iterator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from evaluation_metrics import mae
import scipy as sc
from pymer4.models import Lmer

from pyafm.custom_logistic import CustomLogistic
from sklearn import feature_extraction, model_selection, metrics


# BKT
def bkt(as_data, skills):
    """
    - **Defaults generic parameters**:
        - num_fits (5) is the number of initialization fits used for the BKT model.
        - defaults (None) is a dictionary that can be used to pass values different than the default ones during initialization.
        - parallel (True) indicates whether the computation will use multi-threading.
        - skills ('.\*') is a regular expression used to indicate the skills the BKT model will be run on.
        - seed (random.randint(0, 1e8)) is a seed that can be setup to enable reproducible experiments.
        - folds (5) is the number of folds used in case of cross-validation.
        - forgets (False) indicates whether the model will consider that the student may give a wrong answer even though they had learned the concept.

    - **Defaults additional parameters**:
        - order_id ('order_id') is the name of the CSV column for the chronological IDs that refer to the original problem log.
        - skill_name ('skill_name') is the name of the CSV column for the skill name associated with the problem.
        - correct ('correct') is the name of the CSV column for the correct / incorrect label on the first attempt.
        - user_id ('user_id') is the name of the CSV column for the ID of the student doing the problem.
        - multilearn ('template_id') is the name of the column for checking whether there is a multi-skill object.
        - multiprior ('correct') is the name of the CSV column for mapping multi-prior knowledge.
        - multigs ('template_id') is the name of the CSV column corresponding to the desired guess/slip classes.

    - **Initializers for learnable parameters**:
        - 'prior' (None, no inizialization) is the initial probability of answering the question correct.
        - 'learns' (None, no inizialization) is the probability that the student has learned something that was previous not known.
        - 'guesses' (None, no inizialization) is the probability that the student guessed the right answer while not knowing the concept.
        - 'slips' (None, no inizialization) is the probability that the student gave a wrong answer even though they had learned the concept.
        - 'forgets' (None, no inizialization) is the probability that the student forgot something previously learned.
    """
    model = Model(parallel=True, num_fits=5, seed=42, defaults=None)

    # fits a BKT model given model and data information.
    # Takes arguments skills, number of initialization fits,
    # default column names (i.e. correct, skill_name), parallelization, and model types.
    model.fit(data=as_data, skills=skills)

    # evaluates a BKT model given model and data information.
    # Takes a metric and data path or DataFrame as arguments.
    # Returns the value of the metric for the given trained model tested on the given data.
    model.evaluate(data=as_data, metric="auc")

    # crossvalidates (trains and evaluates) the BKT model.
    # Takes the data, metric, and any arguments that would be
    #  passed to the fit function (skills, number of initialization
    # fits, default column names, parallelization, and model types) as arguments.
    model.crossvalidate(data=as_data, metric="auc", folds=5)

    # view learned parameters of the BKT model.
    # prior ( 𝑃0)
    #   the prior probability of "knowing".
    # forgets ( 𝑃F)
    #  : the probability of transitioning to the "not knowing" state given "known".
    # learns ( 𝑃L
    #  ): the probability of transitioning to the "knowing" state given "not known".
    # slips ( 𝑃S
    #  ): the probability of picking incorrect answer, given "knowing" state.
    # guesses ( 𝑃G
    #  ): the probability of guessing correctly, given "not knowing" state.
    model.coef_
    model.params()

    # specify param:
    # model.coef_ = {'Box and Whisker': {'prior': 1e-40}}

    # train a multiguess and slip BKT model on the same skills in the data set.
    # The multigs model fits a different guess/slip rate for each class.
    # Note that, with multigs=True, the guess and slip classes will be
    #  specified by the template_id. You can specify a custom column
    #  mapping by doing multigs='column_name'.
    model.fit(data=as_data, skills=skills, multigs=True)

    #  multilearn model fits a different learn rate (and forget rate if enabled)
    #  rate for each class specified. Note that, with multilearn=True, the learn
    #  classes are specified by the template_id. You can specify a custom column
    #  mapping by doing multilearn='column_name'.
    model.fit(data=as_data, skills=skills, multilearn=True)

    # interested in learning the parameters for each student, and we also enable forgetting
    model.fit(data=as_data, skills=skills, forgets=True, multilearn="user_id")

    # Make predictions
    preds = model.predict(data=as_data)

    # custom metrics
    model.evaluate(data=as_data, metric=mae)
    model.evaluate(data=as_data, metric="rmse")


def train_BKT_per_skill(data, skill_data, skills_subset):
    rmse_bkt, auc_bkt = [], []
    df_preds = pd.DataFrame()
    # Train a BKT model for each skill
    for skill in skills_subset:
        print("--", skill, "--")
        skill_data = data[data["skill_name"] == skill]
        for iteration, (train_index, test_index) in enumerate(
            create_iterator(skill_data)
        ):
            # Split data in training and test sets
            X_train, X_test = skill_data.iloc[train_index], skill_data.iloc[test_index]
            # Initialize and fit the model
            model = Model(seed=0)
            model.fit(data=X_train)
            # Compute predictions
            preds = model.predict(data=X_test)[
                ["user_id", "skill_name", "correct", "correct_predictions"]
            ]
            df_preds = df_preds.append(preds)

    return df_preds


def evaluate_bkt(df_preds):
    # Compute overall RMSE and AUC
    rmse = mean_squared_error(
        df_preds.correct, df_preds.correct_predictions, squared=False
    )
    AUC = roc_auc_score(df_preds.correct, df_preds.correct_predictions)
    print("Overall RMSE:", rmse, "Overall AUC:", AUC)

    # Compute RMSE and AUC per skill
    skill_rmse = df_preds.groupby(["skill_name"]).apply(
        lambda df_preds: mean_squared_error(
            df_preds.correct, df_preds.correct_predictions, squared=False
        )
    )
    print("RMSE", np.mean(skill_rmse), np.std(skill_rmse))

    skill_auc = df_preds.groupby(["skill_name"]).apply(
        lambda df_preds: roc_auc_score(df_preds.correct, df_preds.correct_predictions)
    )
    print("AUC", np.mean(skill_auc), np.std(skill_auc))


def plot_bkt_evaluation_per_skill(skill_rmse, skill_auc, skills_subset):
    # Create overall RMSE and RMSE per skill data frames
    skills_all = ["Skills"] * len(skills_subset)
    df_overall_rmse = pd.DataFrame(
        list(zip(skills_all, skill_rmse)), columns=["x", "RMSE"]
    )
    df_skill_rmse = pd.DataFrame(
        list(zip(skills_subset, skill_rmse)), columns=["x", "RMSE"]
    )

    # Create overall AUC and AUC per skill data frames
    df_overall_auc = pd.DataFrame(
        list(zip(skills_all, skill_auc)), columns=["x", "AUC"]
    )
    df_skill_auc = pd.DataFrame(
        list(zip(skills_subset, skill_auc)), columns=["x", "AUC"]
    )

    # Two bar plots for RMSE: first one with std (mean RMSE over all skills), then one bar plot with a bar for each specific skill
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(
        ax=axes[0], x="x", y="RMSE", data=df_overall_rmse, estimator=np.mean, ci="sd"
    )
    axes[0].set_title("Overall RMSE")
    axes[0].set_xlabel("")

    sns.barplot(
        ax=axes[1], x="x", y="RMSE", data=df_skill_rmse, estimator=np.mean, ci="sd"
    )
    plt.xticks(rotation=90)
    axes[1].set_title("RMSE across skills")
    axes[1].set_xlabel("")

    fig.show()

    # Two bar plots for AUC: first one with std (mean AUC over all skills), then one bar plot with a bar for each specific skill
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(
        ax=axes[0], x="x", y="AUC", data=df_overall_auc, estimator=np.mean, ci="sd"
    )
    axes[0].set_title("Overall AUC")
    axes[0].set_xlabel("")

    sns.barplot(
        ax=axes[1], x="x", y="AUC", data=df_skill_auc, estimator=np.mean, ci="sd"
    )
    plt.xticks(rotation=90)
    axes[1].set_title("AUC across skills")
    axes[1].set_xlabel("")

    fig.show()


# Learning curves
def avg_y_by_x(x, y):
    """
    Compute average learning curve and number of students over the number of opportunities.
    x is the number of opportunities.
    y the success rates of the users (can be predicted success rate or true success rate).
    """
    # Transform lists into arrays
    x = np.array(x)
    y = np.array(y)

    # Sort the integer id representing the number of opportunities in increasing order
    xs = sorted(list(set(x)))

    # Supporting lists to store the:
    # - xv: integer identifier of the number of opportunities
    # - yv: average value across students at that number of opportunities
    # - lcb and ucb: lower and upper confidence bound
    # - n_obs: number of observartions present at that number of opportunities (on per-skill plots, it is the #students)
    xv, yv, lcb, ucb, n_obs = [], [], [], [], []

    # For each integer identifier of the number of opportunities 0, ...
    for v in xs:
        ys = [
            y[i] for i, e in enumerate(x) if e == v
        ]  # We retrieve the values for that integer identifier
        if len(ys) > 0:
            xv.append(v)  # Append the integer identifier of the number of opportunities
            yv.append(
                sum(ys) / len(ys)
            )  # Append the average value across students at that number of opportunities
            n_obs.append(
                len(ys)
            )  # Append the number of observartions present at that number of opportunities

            # Prepare data for confidence interval computation
            unique, counts = np.unique(ys, return_counts=True)
            counts = dict(zip(unique, counts))

            if 0 not in counts:
                counts[0] = 0
            if 1 not in counts:
                counts[1] = 0

            # Calculate the 95% confidence intervals
            ci = sc.stats.beta.interval(0.95, 0.5 + counts[0], 0.5 + counts[1])
            lcb.append(ci[0])
            ucb.append(ci[1])

    return xv, yv, lcb, ucb, n_obs


def plot_learning_curve(skill_name, predictions):
    """
    Plot learning curve using BKT model for skill `skill_name`.
    """
    preds = predictions[
        predictions["skill_name"] == skill_name
    ]  # Retrieve predictions for the current skill

    xp = []
    yp = {}
    for (
        col
    ) in (
        preds.columns
    ):  # For y_true and and y_pred_bkt columns, initialize an empty list for curve values
        if "y_" in col:
            yp[col] = []

    for user_id in preds["user_id"].unique():  # For each user
        user_preds = preds[
            preds["user_id"] == user_id
        ]  # Retrieve the predictions on the current skill for this user
        xp += list(
            np.arange(len(user_preds))
        )  # The x-axis values go from 0 to |n_opportunities|-1
        for col in preds.columns:
            if "y_" in col:  # For y_true and and y_pred_bkt columns
                yp[col] += user_preds[
                    col
                ].tolist()  # The y-axis value is the success rate for this user at that opportunity

    fig, axs = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [3, 2]}
    )  # Initialize the plotting figure

    lines = []
    for col in preds.columns:
        if "y_" in col:  # For y_true and and y_pred_bkt columns
            x, y, lcb, ucb, n_obs = avg_y_by_x(
                xp, yp[col]
            )  # Calculate mean and 95% confidence intervals for success rate
            y = [1 - v for v in y]  # Transform success rate in error rate
            if (
                col == "y_true"
            ):  # In case of ground-truth data, we also show the confidence intervals
                axs[0].fill_between(x, lcb, ucb, alpha=0.1)
            (model_line,) = axs[0].plot(x, y, label=col)  # Plot the curve
            lines.append(model_line)  # Store the line to then set the legend

    # Make decorations for the learning curve plot
    axs[0].set_title(skill_name)
    axs[0].legend(handles=lines)
    axs[0].set_ylabel("Error")
    axs[0].set_ylim(0, 1)
    axs[0].set_xlim(0, None)

    # Plot the number of observations per number of opportunities bars and make decorations
    axs[1].set_xlabel("#Opportunities")
    axs[1].bar([i for i in range(len(n_obs))], n_obs)
    axs[1].set_ylabel("#Observations")
    axs[1].set_ylim(0, 750)
    axs[1].set_xlim(0, None)

    # Plot the learning curve and the bar plot
    plt.show()


# AFM
def afm_model(X_train, X_test):
    """
    only works for a one-to-one correspondance of task and skill,
    i.e. when a task is associated to exactly one skill.
    In case of a data set containing tasks with multiple skills,
    we would need to use the pyAFM package.
    A tutorial on using pyAFM can be found here:
    https://github.com/epfl-ml4ed/mlbd-2021/tree/main/Tutorials/Tutorial06/Tutorial06
    """
    model = Lmer(
        "correct ~ (1|user_id) + (1|skill_name) + (0 + prev_attempts|skill_name)",
        data=X_train,
        family="binomial",
    )
    model.fit()
    afm_preds = model.predict(X_test, verify_predictions=False)


# AFM with pyAFM
def afm_pyafm_model(X_train, y_train, X_test, y_test):
    """(e.g., no custom bounds, default l2 regularization, and fit_intercept=True)"""
    afm = CustomLogistic()
    afm.fit(X_train, y_train)
    y_test_pred = afm.predict_proba(X_test)
    test_rmse = metrics.mean_squared_error(y_test, y_test_pred, squared=False)


# Pyafm helper for formatting
def read_as_student_step(data):
    skills, opportunities, corrects, user_ids = [], [], [], []

    for row_id, (_, row) in enumerate(data.iterrows()):
        # Get attributes for the current interaction
        user_id = row["user_id"]
        skill_name = row["skill_name"]
        correct = row["correct"]
        prior_success = row["prior_success"]
        prior_failure = row["prior_failure"]

        # Update the number of opportunities this student had with this skill
        opportunities.append({skill_name: prior_success + prior_failure})

        # Update information in the current
        skills.append({skill_name: 1})

        # Answer info
        corrects.append(correct)

        # Student info
        user_ids.append({user_id: 1})

    return (skills, opportunities, corrects, user_ids)


def prepare_data_afm(skills, opportunities, corrects, user_ids):
    sv = feature_extraction.DictVectorizer()
    qv = feature_extraction.DictVectorizer()
    ov = feature_extraction.DictVectorizer()
    S = sv.fit_transform(user_ids)
    Q = qv.fit_transform(skills)
    O = ov.fit_transform(opportunities)
    X = sc.sparse.hstack((S, Q, O))
    y = np.array(corrects)

    return (X.toarray(), y)


def pfa_model(X_train, X_test):
    model = Lmer(
        "correct ~ (1|user_id) + (1|skill_name) + (0 + before_correct_num|skill_name) + (0 + before_wrong_num|skill_name)",
        data=X_train,
        family="binomial",
    )
    model.fit()
    # Compute predictions
    pfa_predictions = model.predict(data=X_test, verify_predictions=False)


# PFA with pyAFM
def pfa_pyafm_model(X_train, y_train, X_test, y_test):
    # Format data for pyAFM
    # n_succ, n_fail = read_as_success_failure(data)
    # X, y = prepare_data_pfa(skills, corrects, user_ids, n_succ, n_fail)

    pfa = CustomLogistic()
    pfa.fit(X_train, y_train)
    y_test_pred = pfa.predict_proba(X_test)
    test_rmse = metrics.mean_squared_error(y_test, y_test_pred, squared=False)


# PFA helpers
def read_as_success_failure(data):
    n_succ, n_fail = [], []

    # Create the n_succ and n_fail variables required by pyAFM
    for i, row in data.iterrows():
        n_succ.append({row["skill_name"]: int(row["prior_success"])})
        n_fail.append({row["skill_name"]: int(row["prior_failure"])})

    return n_succ, n_fail


def prepare_data_pfa(skills, corrects, user_ids, n_succ, n_fail):
    s = feature_extraction.DictVectorizer()
    q = feature_extraction.DictVectorizer()
    succ = feature_extraction.DictVectorizer()
    fail = feature_extraction.DictVectorizer()
    S = s.fit_transform(user_ids)
    Q = q.fit_transform(skills)
    succ = succ.fit_transform(n_succ)
    fail = fail.fit_transform(n_fail)
    X = sc.sparse.hstack((S, Q, succ, fail))
    y = np.array(corrects)

    return (X.toarray(), y)


if __name__ == "__main__":
    DATA_DIR = "data/"
    assistments = pd.read_csv(DATA_DIR + "assistments.csv", low_memory=False).dropna()
    skills_subset = [
        "Circle Graph",
        "Venn Diagram",
        "Mode",
        "Division Fractions",
        "Finding Percents",
        "Area Rectangle",
    ]
    data = assistments[assistments["skill_name"].isin(skills_subset)]
    model = Model(seed=0)
    model.fit(data=data)
    predictions = model.predict(data=data)[
        ["user_id", "skill_name", "correct", "correct_predictions"]
    ]
    predictions.columns = ["user_id", "skill_name", "y_true", "y_pred_bkt"]

    plot_learning_curve("Circle Graph", predictions)

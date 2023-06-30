# 03

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Import the linear regression model class
from pymer4.models import Lm

# Import the lmm model class
from pymer4.models import Lmer

# Import Gaussian modeling
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Data directory
DATA_DIR = "data"


# Standardize data
def z_score(x):
    return (x - np.mean(x)) / np.std(x)


def min_max(x):
    """normalizing between 0 and 1"""
    # return (x - np.min(x)) / (np.max(x) - np.min(x))

    scaler = MinMaxScaler()
    scaler.fit((x).to_numpy().reshape(-1, 1))
    return scaler.transform((x).to_numpy().reshape(-1, 1))


def linear_model(df, x, y, x2=None, group=None):
    # y is continuous, e.g. "grade"
    # default family is Gaussian - dependent feature (y) is continuous (not discrete or binary)
    # default link is identity
    # default y intercept is True
    model = Lm(f"{y} ~ {x}", data=df)

    # no intercept
    Lm(f"{y} ~ 0 + {x} ", data=df, family="gaussian")

    # two vars
    Lm(f"{y} ~ {x} + {x2}", data=df)

    # Logistic Regression: y is binary, e.g. "passed"
    smf.glm(formula=f"{y} ~ {x}", data=df, family=sm.families.Binomial()).fit()

    # Poisson Regression: y is count, descrete, non-negative, e.g. "number of awards"
    smf.glm(formula=f"{y} ~ {x}", data=df, family=sm.families.Poisson()).fit()

    # mixed effect models
    # Fixed effects: variables that are manipulated by the experimenter
    # Group fixed effects allow us to difference out any constant differences between groups,
    # and focus only on changes within each entity over time.
    Lm(f"{y} ~ 1 + {x}  + {group}", data=df, family="gaussian")

    # Random effects: variables that are not manipulated by the experimenter
    # Initialize model instance using 1 predictor with random intercepts and slopes
    # different y-intercept per group
    Lmer(f"{y} ~ 1 + {x} + (1|{group})", data=df, family="binomial")
    # different slope per group
    Lmer(f"{y} ~ 1 + (0 + {x}|{group})", data=df, family="gaussian")
    # different y-intercept and slope per group
    Lmer(f"{y} ~ (1 + {x}|{group})", data=df, family="gaussian")
    # random intercept and slope for groups AND interaction between the x and time (weeks)
    Lmer(f"{y} ~  (1 + {x}*week|{group})", data=df, family="gaussian(link='log')")

    # Mixed Effect Cheat sheet
    # https://eshinjolly.com/pymer4/rfx_cheatsheet.html
    # Random intercepts only
    # (1 | Group)

    # Random slopes only
    # (0 + Variable | Group)

    # Random intercepts and slopes (and their correlation)
    # (Variable | Group)
    # (1 + Variable | Group)

    # Random intercepts and slopes (without their correlation)
    # (1 | Group) + (0 + Variable | Group)

    # Same as above but will not separate factors (see: https://rdrr.io/cran/lme4/man/expandDoubleVerts.html)
    # (Variable || Group)

    # Random intercept and slope for more than one variable (and their correlations)
    # (Variable_1 + Variable_2 | Group)

    # Conclusion
    # Effects are fixed if they are interesting in themselves
    # or random if there is interest in the underlying population.
    # With intercept random effects, we assumed that every group has a different starting
    # point (y-intercept) and with slope random effects we assume that every group has a different rate.

    model.fit()
    print(model.coefs)
    return model


def plot_lm(model, x):
    model.plot(x, plot_ci=True)

    # or

    intercept = model.coefs.Estimate[0]
    model.ranef.head()
    x = np.linspace(0, 3, 4)
    for i, row in model.ranef.iterrows():
        sns.lineplot(x=x, y=intercept + row["studying_hours"] * x)


def graph_line_best_fit(df, x, y, model):
    x_pred = df[x]
    y_pred = model.fits

    plt.figure()
    plt.scatter(df[x], df[y])
    plt.plot(x_pred, y_pred, color="red")
    plt.xlabel("Mean playback speed")
    plt.ylabel("Grade")
    plt.show()


def r_squared(df, x, y, model):
    """fraction of explained variability of the data
    want it to be as close to 1
    when goal is interpretation
    """
    y_pred = model.fits
    y_actual = df[y]
    return mean_squared_error(y_actual, y_pred)


def mae(df, x, y, model):
    """mean absolute error
    when goal is prediction
    """
    y_pred = model.fits
    y_actual = df[y]
    return np.mean(np.abs(y_actual - y_pred))


def rmse(df, x, y, model):
    """root mean squared error
    when goal is prediction
    penalizes large errors heavier
    """
    y_pred = model.fits
    y_actual = df[y]
    return np.sqrt(mean_squared_error(y_actual, y_pred))


if __name__ == "__main__":
    df = pd.read_csv("{}/aggregated_extended_fc.csv".format(DATA_DIR))
    df = df.fillna("NaN")

    model = linear_model(df, x="mu_speed_playback_mean", y="grade")

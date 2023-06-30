import numpy as np
import matplotlib.pyplot as plt

from lime import lime_tabular
import shap


def partial_dependency_plot(model, features):
    # We generate the PDP plot against a background distribution of all the points available in the feature set.
    # While a minimal background distribution would let us run this analysis faster (i.e. 300 points), we recommend
    # plotting with a much larger point distribution (all the students) if you use this in other situations for
    # improved accuracy and a more global understanding of your model's behavior.

    background_distribution = features

    # This function converts our data to the right format for the PDP explainer.
    predict_fn = lambda x: (1 - model.predict(x)).flatten()  # noqa E731

    # Let's examine the PDP for TotalTimeProblem_InWeek2.
    feat_1 = list(features.columns).index("TotalTimeProblem_InWeek2")

    # Create a partial dependence plot from the background distribution.
    fig = shap.plots.partial_dependence(
        feat_1,
        predict_fn,
        background_distribution,
        ice=False,
        ylabel="Model Predictions",
        model_expected_value=True,
        feature_expected_value=False,
        show=True,
    )


def lime_explainability(model, features, class_names=["pass", "fail"]):
    # This function returns a (NUM OF INSTANCES, 2) array of probability
    # of pass in first column and probability of failing in another
    # column, which is the format LIME requires.
    predict_fn = (
        lambda x: np.array([[1 - model.predict(x)], [model.predict(x)]])
        .reshape(2, -1)
        .T
    )

    # We initialize the LIME explainer on our training data.
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(features),
        feature_names=features.columns,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
    )

    # We choose to explain the behavior of student 1.
    instance = 1

    # This line calls our LIME explainer on a student instance.
    exp_instance = explainer.explain_instance(
        features.iloc[instance], predict_fn, num_features=10
    )


# Let's plot the results.
def plot_lime(exp, instance, features, labels, loaded_model):
    s = "fail" if labels[instance] else "pass"
    label = exp.available_labels()[0]
    expl = exp.as_list(label=label)
    fig = plt.figure(facecolor="white")
    vals = [x[1] for x in expl]
    names = [x[0] for x in expl]
    vals.reverse()
    names.reverse()
    colors = ["green" if x > 0 else "red" for x in vals]
    pos = np.arange(len(expl)) + 0.5
    plt.barh(pos, vals, align="center", color=colors)
    plt.yticks(pos, names)
    prediction = loaded_model.predict(
        np.array(features.iloc[instance]).reshape(1, 250)
    )[0][0]
    prediction = np.round(1 - prediction, 2)
    print("Student #: ", instance)
    print("Ground Truth Model Prediction: ", 1 - labels[instance], "-", s)
    print(
        "Black Box Model Prediction: ",
        prediction,
        "-",
        "pass" if prediction > 0.5 else "fail",
    )

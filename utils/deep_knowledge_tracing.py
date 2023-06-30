# 08
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

# In this demo, we use a lot of SciKit-Learn functions, as imported below.
from sklearn.model_selection import ParameterGrid, train_test_split

from preprocessing import create_iterator


def prepare_seq(df):
    """
    Extract user_id sequence in preparation for DKT. The output of this function
    feeds into the prepare_data() function.
    """
    # Enumerate skill id as a categorical variable
    # (i.e. [32, 12, 32, 45] -> [0, 1, 0, 2])
    df["skill"], skill_codes = pd.factorize(df["skill_name"], sort=True)

    # Cross skill id with answer to form a synthetic feature
    df["skill_with_answer"] = df["skill"] * 2 + df["correct"]

    # Convert to a sequence per user_id and shift features 1 timestep
    seq = df.groupby("user_id").apply(
        lambda r: (
            r["skill_with_answer"].values[:-1],
            r["skill"].values[1:],
            r["correct"].values[1:],
        )
    )

    # Get max skill depth and max feature depth
    skill_depth = df["skill"].max() + 1
    features_depth = df["skill_with_answer"].max() + 1

    return seq, features_depth, skill_depth


def prepare_data(seq, params, features_depth, skill_depth):
    """
    Manipulate the data sequences into the right format for DKT with padding by batch
    and encoding categorical features.
    """

    # Get Tensorflow Dataset
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: seq, output_types=(tf.int32, tf.int32, tf.float32)
    )

    # Encode categorical features and merge skills with labels to compute target loss
    dataset = dataset.map(
        lambda feat, skill, label: (
            tf.one_hot(feat, depth=features_depth),
            tf.concat(
                values=[
                    tf.one_hot(skill, depth=skill_depth),
                    tf.expand_dims(label, -1),
                ],
                axis=-1,
            ),
        )
    )

    # Pad sequences to the appropriate length per batch
    dataset = dataset.padded_batch(
        batch_size=params["batch_size"],
        padding_values=(params["mask_value"], params["mask_value"]),
        padded_shapes=([None, None], [None, None]),
        drop_remainder=True,
    )

    return dataset.repeat(), len(seq)


def tf_datasets(data, params):
    # Obtain indexes for training and test sets
    train_index, test_index = next(create_iterator(data))

    # Split the data into training and test
    X_train, X_test = data.iloc[train_index], data.iloc[test_index]

    # Obtain indexes for training and validation sets
    train_val_index, val_index = next(create_iterator(X_train))

    # Split the training data into training and validation
    X_train_val, X_val = X_train.iloc[train_val_index], X_train.iloc[val_index]

    # Build TensorFlow sequence datasets for training, validation, and test data
    seq, features_depth, skill_depth = prepare_seq(data)
    seq_train = seq[X_train_val.user_id.unique()]
    seq_val = seq[X_val.user_id.unique()]
    seq_test = seq[X_test.user_id.unique()]

    # Prepare the training, validation, and test data in the DKT input format
    tf_train, length = prepare_data(seq_train, params, features_depth, skill_depth)
    tf_val, val_length = prepare_data(seq_val, params, features_depth, skill_depth)
    tf_test, test_length = prepare_data(seq_test, params, features_depth, skill_depth)

    # Calculate the length of each of the train-test-val sets and store as parameters
    params["train_size"] = int(length // params["batch_size"])
    params["val_size"] = int(val_length // params["batch_size"])
    params["test_size"] = int(test_length // params["batch_size"])

    return features_depth, skill_depth, params, tf_train, tf_val, tf_test


def get_params():
    # Specify the model hyperparameters
    params = {}

    # The 'batch_size' parameter refers to the number of instances the model evaluates
    # at a time. We choose the 'batch_size' based on the size of our dataset, the diversity
    # of our data instances, and the size of each instance. Most times, people use batches
    # in the binary family tree {8, 16, 32, 64, 128, 256}. If your batch size is large, your
    # model trains faster, but if your batch size is small, each instance is more valued in
    # the model training process. It's a tradeoff between computational efficiency and
    # instance importance.
    params["batch_size"] = 32

    # The parameter 'mask_value' tells our model which input values to ignore. None of
    # our features will have the value -1.0 naturally, so it's a good choice to use for
    # marking (or 'masking') invalid values so the model does not consider them. We will
    # use this parameter in our padding case, since we don't want our model to consider the
    # extra values we're including in our sequences to make them the same length in each batch.
    params["mask_value"] = -1.0

    # The 'verbose' parameter can be set to {0, 1, 2} to specify the level of logging we want from
    # the model during training. 1 is usually a good compromise between being inundated with info
    # and not knowing what's going on in your model (i.e. loss, epochs, metric).
    params["verbose"] = 1

    # As we train the model, we want to continually save only the best model iteration.
    # The 'best_model_weights' parameter tells our training code where to save the model.
    params["best_model_weights"] = "weights/bestmodel"

    # The 'optimizer' parameter specifies which gradient descent optimizer to use in backpropogation.
    # 'adam' is a common choice. Others include 'SGD', 'RMSProp', or 'ADAgrad', among many others.
    params["optimizer"] = "adam"

    # The 'recurrent_units' parameter refers to the of hidden units in your recurrent layer.
    # This is a very common hyperparameter to tune as it usually makes a substantial difference
    # in model performance (as we'll see later on).
    params["recurrent_units"] = 16

    # How long do we want our model to train? Each time the model sees all the data in the training
    # process is an 'epoch'. If the model trains for too long (sees the data too many times), it'll
    # overfit; although we don't have to worry about this because we're only saving the best model
    # (see: best_model_weights). If it trains for not long enough, it'll underfit. There's also a
    # tradeoff between quality of model and amount of computational resources / time. For now, 20's
    # a good number. As a rule of thumb, 100 epochs is usually a lot, and 10 epochs is usually too
    # little, but it depends on the diversity of your data and what you're trying to predict.
    params["epochs"] = 20

    # A really interesting research discovery in 2013 found that neural nets perform better when
    # they forget a portion of their weights and replace them with 0s. This is the notion of dropout.
    # The 'dropout_rate' parameter tells our model what fraction of weights to throw away or 'drop out'
    # each epoch. Want to read the paper? https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
    params["dropout_rate"] = 0.1


def get_target(y_true, y_pred, mask_value=-1.0):
    """
    Considering that we padded the sequences such that all have the same
    length, we need to remove predictions for the time steps that are
    based on padded data
    Adjust y_true and y_pred to ignore predictions made using padded values.
    """
    # Get skills and labels from y_true
    mask = 1.0 - tf.cast(tf.equal(y_true, mask_value), y_true.dtype)
    y_true = y_true * mask

    skills, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)

    # Get predictions for each skill
    y_pred = tf.reduce_sum(y_pred * skills, axis=-1, keepdims=True)

    return y_true, y_pred


class AUC(tf.keras.metrics.AUC):
    # Our custom AUC calls our get_target function first to remove predictions on padded values,
    # then computes a standard AUC metric.
    def __init__(self):
        # We use a super constructor here just to make our metric name pretty!
        super(AUC, self).__init__(name="auc")

    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        super(AUC, self).update_state(
            y_true=true, y_pred=pred, sample_weight=sample_weight
        )


class RMSE(tf.keras.metrics.RootMeanSquaredError):
    # Our custom RMSE calls our get_target function first to remove predictions on padded values,
    # then computes a standard RMSE metric.
    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        super(RMSE, self).update_state(
            y_true=true, y_pred=pred, sample_weight=sample_weight
        )


def CustomBinaryCrossEntropy(y_true, y_pred):
    # Our custom binary cross entropy loss calls our get_target function first
    # to remove predictions on padded values, then computes standard binary cross-entropy.
    y_true, y_pred = get_target(y_true, y_pred)
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)


def create_model_lstm(nb_features, nb_skills, params):
    # Create an LSTM model architecture
    inputs = tf.keras.Input(shape=(None, nb_features), name="inputs")

    # We use a masking layer here to ignore our masked padding values
    x = tf.keras.layers.Masking(mask_value=params["mask_value"])(inputs)

    # This LSTM layer is the crux of the model; we use our parameters to specify
    # what this layer should look like (# of recurrent_units, fraction of dropout).
    x = tf.keras.layers.LSTM(
        params["recurrent_units"], return_sequences=True, dropout=params["dropout_rate"]
    )(
        x
    )  # Binary: return_sequences=False when we want a many-to-one architecture
    # When return_sequences is false, we don't use a TimeDistributed layer

    # We use a dense layer with the sigmoid function activation to map our predictions
    # sigmoid: between 0 and 1 for binary classification
    # linear:  on a linear scale
    dense = tf.keras.layers.Dense(
        nb_skills, activation="sigmoid"
    )  # activation='linear'

    # The TimeDistributed layer takes the dense layer predictions and applies the sigmoid
    # activation function to all time steps.
    outputs = tf.keras.layers.TimeDistributed(dense, name="outputs")(x)
    # outputs = dense(x) # when return_sequences is false
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="DKT")

    # Compile the model with our loss functions, optimizer, and metrics.
    model.compile(
        loss=CustomBinaryCrossEntropy,  # loss=tf.keras.losses.MSE, tf.keras.losses.binary_crossentropy
        optimizer=params["optimizer"],
        metrics=[AUC(), RMSE()],  # Binary: tf.keras.metrics.AUC(), 'binary_accuracy']
    )

    return model


def create_model_lstm_predict_score_sigmoid(nb_features, nb_skills, params):
    inputs = tf.keras.Input(shape=(None, nb_features), name="inputs")
    x = tf.keras.layers.Masking(mask_value=params["mask_value"])(inputs)
    x = tf.keras.layers.LSTM(
        params["recurrent_units"], return_sequences=True, dropout=params["dropout_rate"]
    )(x)
    dense = tf.keras.layers.Dense(
        nb_skills, activation="sigmoid"
    )  # activation='linear'
    outputs = tf.keras.layers.TimeDistributed(dense, name="outputs")(x)
    # restrict predictions between 0 and 10
    outputs = tf.keras.layers.Lambda(lambda x: x * 10)(outputs)

    model = tf.keras.models.Model(
        inputs=inputs, outputs=outputs, name="LSTM_predict_score"
    )


def create_model_lstm_predict_score_linear(nb_features, nb_skills, params):
    inputs = tf.keras.Input(shape=(None, nb_features), name="inputs")
    x = tf.keras.layers.Masking(mask_value=params["mask_value"])(inputs)
    x = tf.keras.layers.LSTM(
        params["recurrent_units"], return_sequences=True, dropout=params["dropout_rate"]
    )(x)
    dense = tf.keras.layers.Dense(nb_skills, activation="linear")
    outputs = tf.keras.layers.TimeDistributed(dense, name="outputs")(x)
    # clip predictions between 0 and 10
    outputs = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, 0, 10))(outputs)

    model = tf.keras.models.Model(
        inputs=inputs, outputs=outputs, name="LSTM_predict_score_linear"
    )


def create_model_gru(nb_features, nb_skills, params):
    # Create a GRU model architecture
    inputs = tf.keras.Input(shape=(None, nb_features), name="inputs")

    # We use a masking layer here to ignore our masked padding values
    x = tf.keras.layers.Masking(mask_value=params["mask_value"])(inputs)

    # This GRU layer is the crux of the model; we use our parameters to specify
    # what this layer should look like (# of recurrent_units, fraction of dropout).
    x = tf.keras.layers.GRU(
        params["recurrent_units"], return_sequences=True, dropout=params["dropout_rate"]
    )(x)

    # We use a dense layer with the sigmoid function activation to map our predictions
    # between 0 and 1.
    dense = tf.keras.layers.Dense(nb_skills, activation="sigmoid")

    # The TimeDistributed layer takes the dense layer predictions and applies the sigmoid
    # activation function to all time steps.
    outputs = tf.keras.layers.TimeDistributed(dense, name="outputs")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="DKT")

    # Compile the model with our loss functions, optimizer, and metrics.
    model.compile(
        loss=CustomBinaryCrossEntropy,
        optimizer=params["optimizer"],
        metrics=[AUC(), RMSE()],
    )

    return model


def train_model(model, tf_train, tf_val, params):
    # This line tells our training procedure to only save the best version of the model at any given time.
    ckp_callback = tf.keras.callbacks.ModelCheckpoint(
        params["best_model_weights"], save_best_only=True, save_weights_only=True
    )

    # Let's fit our LSTM model on the training data. This cell takes 8 minutes to run on Colab.
    history = model.fit(
        tf_train,  # (df_x_train_val, df_y_train_val)
        epochs=params["epochs"],
        steps_per_epoch=params["train_size"] - 1,
        validation_data=tf_val,
        validation_steps=params["val_size"],
        callbacks=[ckp_callback],
        verbose=params["verbose"],
    )


def evaluate_model(model, tf_test, params):
    model.load_weights(params["best_model_weights"])
    model.evaluate(
        tf_test, steps=params["test_size"], verbose=params["verbose"], return_dict=True
    )


def hyperparam_tuning_lstm(params, features_depth, skill_depth):
    # Modify the dictionary of parameters so that each parameter maps to a list of possibilities.
    # In this case, we're only searching over the recurrent_units and leaving the rest of the
    # parameters fixed to their default values.
    params_space = {param: [value] for param, value in params.items()}
    params_space["recurrent_units"] = [8, 16, 32, 64]
    params_grid = ParameterGrid(params_space)

    # For each combination of candidate parameters, fit a model on the training set
    # and evaluate it on the validation set (as we've seen in Lecture 5).

    # NOTE: This cell will take 40 minutes to run.
    results = {}

    # For each parameter setting in the grid search of parameters
    for params_i in params_grid:
        # Create a LSTM model with the specific parameter setting params_i
        dkt_lstm = create_model_lstm(features_depth, skill_depth, params_i)

        save_model_name = params_i["best_model_weights"] + str(
            params_i["recurrent_units"]
        )

        # Save the best version of the model through the training epochs
        ckp_callback = tf.keras.callbacks.ModelCheckpoint(
            save_model_name, save_best_only=True, save_weights_only=True
        )

        # Fit the model on the training data with the appropriate parameters
        dkt_lstm.fit(
            tf_train,
            epochs=params_i["epochs"],
            steps_per_epoch=params_i["train_size"] - 1,
            validation_data=tf_val,
            validation_steps=params_i["val_size"],
            callbacks=[ckp_callback],
            verbose=params_i["verbose"],
        )

        # Evaluate the model performance
        results[params_i["recurrent_units"]] = dkt_lstm.evaluate(
            tf_val,
            steps=params_i["val_size"],
            verbose=params_i["verbose"],
            return_dict=True,
        )

    # Sort candidate parameters according to their accuracy
    results = sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True)

    # Obtain the best parameters
    best_params = results[0][0]
    print(best_params)

    # Load the best model variant from the hyperparameter gridsearch
    dkt_lstm.load_weights(params["best_model_weights"] + str(best_params))
    dkt_lstm.evaluate(
        tf_test, steps=params["test_size"], verbose=params["verbose"], return_dict=True
    )


def mask_missing_answers(df, num_features, y_column="quiz_correct", time_column="week"):
    num_index = df.shape[1] - num_features
    # Mask df_x values
    mask = df[y_column].isna().values
    mask = np.concatenate(
        [
            np.zeros((mask.shape[0], num_index), dtype=bool),
            mask[:, None].repeat(num_features, axis=1),
        ],
        axis=1,
    )
    df_x = df.mask(mask, -1)
    df_y = df.fillna(-1)

    # Reshape df_x and df_y
    num_weeks = df_y[time_column].nunique()
    df_y = df_y[y_column].values.reshape(-1, num_weeks, 1)
    df_x = df_x.iloc[:, num_index:].values.reshape(-1, num_weeks, num_features)

    return df_x, df_y


def binary_prediction_after_n_weeks(
    df_x, df_binary_labels, binary_column="label-pass-fail", n=1
):
    df_x_binary = df_x[:, :n, :]
    df_y_binary = df_binary_labels[binary_column].values.reshape(-1, 1)

    # Split into training and test sets.
    (
        df_x_binary_train,
        df_x_binary_test,
        df_y_binary_train,
        df_y_binary_test,
    ) = train_test_split(
        df_x_binary, df_y_binary, test_size=0.2, random_state=0, stratify=df_y_binary
    )

    # Split training into training and validation sets.
    (
        df_x_binary_train_val,
        df_x_binary_val,
        df_y_binary_train_val,
        df_y_binary_val,
    ) = train_test_split(
        df_x_binary_train,
        df_y_binary_train,
        test_size=0.2,
        random_state=0,
        stratify=df_y_binary_train,
    )


def build_binary_prediction_data(
    df, y_col, index_col, num_features=1, num_weeks=5, n=5
):
    df_x = df.drop(columns=[y_col])
    df_y = df[[index_col, y_col]]

    # check for nans
    # df_x.isna().sum()
    # df_y.isna().sum()

    # mask
    num_index = df_x.shape[1] - num_features
    mask = df_y.group.isna().values
    mask = np.concatenate(
        [
            np.zeros((mask.shape[0], num_index), dtype=bool),
            mask[:, None].repeat(num_features, axis=1),
        ],
        axis=1,
    )
    df_x = df_x.mask(mask, -1)
    df_y = df_y.fillna(-1)

    # reshape
    df_x = df_x.iloc[:, num_index:].values.reshape(-1, num_weeks, num_features)
    # limit to first n weeks
    df_x = df_x[:, :n, :]

    df_y = df_y["group"].values.reshape(-1, 1)

    # Split the data into training and test set
    df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(
        df_x, df_y, test_size=0.2, random_state=0, stratify=df_y
    )
    # Split the training data further into training and validation sets.
    df_x_train_val, df_x_val, df_y_train_val, df_y_val = train_test_split(
        df_x_train, df_y_train, test_size=0.2, random_state=0, stratify=df_y_train
    )

    # one hot encode y values for categorical
    df_y_train_val = pd.get_dummies(df_y_train_val.flatten()).values
    df_y_val = pd.get_dummies(df_y_val.flatten()).values
    df_y_test = pd.get_dummies(df_y_test.flatten()).values


def one_hot_encode(df_y):
    # one hot encode categories:
    df_y_one_hot = pd.get_dummies(df_y)
    # or
    from keras.utils.np_utils import to_categorical

    df_y_one_hot = to_categorical(df_y.group.astype("category").cat.codes)
    # or
    df_y_ints = df_y.replace(
        {"intervene": 0, "on-track": 1, "advanced": 2}
    ).values.reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(df_y_ints)
    df_y_one_hot = enc.transform(df_y_ints).toarray()


if __name__ == "__main__":
    DATA_DIR = "data/"
    data = pd.read_csv(DATA_DIR + "assistments.csv", low_memory=False).dropna()

    params = get_params()
    features_depth, skill_depth, params, tf_train, tf_val, tf_test = tf_datasets(
        data, params
    )

    dkt_lstm = create_model_lstm(features_depth, skill_depth, params)
    dkt_gru = create_model_gru(features_depth, skill_depth, params)

    train_model(dkt_lstm, tf_train, tf_val, params)
    evaluate_model(dkt_lstm, tf_test, params)

import numpy as np
import tensorflow as tf


def binary_cross_entropy(y_pred, y_true):
    # return nn.BCELoss()
    # return F.binary_cross_entropy(y_pred, y_true)
    return (1 / len(y_pred)) * np.sum(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )


# Custom metrics for training and testing
class RMSE(tf.keras.metrics.RootMeanSquaredError):
    # Our custom RMSE calls our get_target function first to remove predictions on padded values,
    # then computes a standard RMSE metric.
    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        super(RMSE, self).update_state(
            y_true=true, y_pred=pred, sample_weight=sample_weight
        )


class CustomAccuracy(tf.keras.metrics.Accuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.keras.utils.nputils.to_categorical(y_true, 3)
        super(CustomAccuracy, self).update_state(
            y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
        )


mse = tf.keras.losses.MeanSquaredError()


def CustomMeanSquaredError(y_true, y_pred):
    # Our custom mean squared error loss calls our get_target function first
    # to remove predictions on padded values, then computes standard binary cross-entropy.
    y_true, y_pred = get_target(y_true, y_pred)
    return mse(y_true, y_pred)


def CustomMAPE(y_true, y_pred):
    y_true, y_pred = get_target(y_true, y_pred)
    return tf.keras.losses.mape()(y_true, y_pred)


# Function for getting the Tensorflow output sequences for the model
def get_target(y_true, y_pred, mask_value=-1.0):
    """
    Adjust y_true and y_pred to ignore predictions made using padded values.
    """
    # Get skills and labels from y_true
    mask = 1.0 - tf.cast(tf.equal(y_true, mask_value), y_true.dtype)
    y_true = y_true * mask

    problems, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)

    # Get predictions for each skill
    y_pred = tf.reduce_sum(y_pred * problems, axis=-1, keepdims=True)

    return y_true, y_pred

import tensorflow as tf
import keras



@keras.saving.register_keras_serializable()
def sum_over_contracts(x):
    return tf.reduce_sum(x, axis=2)

@keras.saving.register_keras_serializable()
def relative_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square((y_true - y_pred) / (y_true + 1e-6)))

@keras.saving.register_keras_serializable()
def weighted_mse(weight = 1):
    return lambda y_true, y_pred: relative_mse(y_true, y_pred) * weight

@keras.saving.register_keras_serializable()
def bounded_mse(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    penalty = 100.0 * tf.reduce_mean(tf.square(tf.maximum(0.0, y_pred - 1.0)) + tf.square(tf.maximum(0.0, 0.0 - y_pred)))
    return mse + penalty

@keras.saving.register_keras_serializable()
def log_mse(y_true, y_pred):
    log_true = tf.math.log(y_true + 1e-6)
    log_pred = tf.math.log(y_pred + 1e-6)
    return tf.reduce_mean(tf.square(log_true - log_pred))

custom_objects = {
    'relative_mse': relative_mse,
    'weighted_mse_2': weighted_mse(weight=2),
    'weighted_mse_3': weighted_mse(weight=3),
    'bounded_mse': bounded_mse,
    'log_mse': log_mse,
    "sum_over_contracts": sum_over_contracts
}
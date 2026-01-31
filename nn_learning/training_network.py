import tensorflow as tf
import numpy as np
from pathlib import Path

def build_parameter_network(T, num_contracts):
    # Input layer
    inputs = tf.keras.layers.Input(shape=(T, num_contracts))

    # LSTM layers to capture temporal dependencies
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(64)(x)

    # tf.keras.layers.Dense layers to map to parameters
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    # Output heads for each parameter
    # Example: Predicting mu, alpha_moneyness, alpha_time, beta, gamma_t, gamma_m, tau, w_volume, etc.
    outputs = {}
    outputs["mu"] = tf.keras.layers.Dense(1, activation="linear", name="mu")(x)
    outputs["alpha_moneyness"] = tf.keras.layers.Dense(1, activation="linear", name="alpha_moneyness")(x)
    outputs["alpha_time"] = tf.keras.layers.Dense(1, activation="linear", name="alpha_time")(x)
    outputs["beta"] = tf.keras.layers.Dense(1, activation="linear", name="beta")(x)
    outputs["gamma_t"] = tf.keras.layers.Dense(1, activation="linear", name="gamma_t")(x)
    outputs["gamma_m"] = tf.keras.layers.Dense(1, activation="linear", name="gamma_m")(x)

    # 4 = size of tau (2 x 2)
    outputs["tau"] = tf.keras.layers.Dense(4, activation="sigmoid", name="tau")(x)  # Reshape later
    outputs["w_volume"] = tf.keras.layers.Dense(1, activation="linear", name="w_volume")(x)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)

if __name__ == "__main__":

    """
    for spy_eod_202303.txt:
    counts: 6642
    strike count: 289
    exp counts: 53
    T: 23
    """

    T = 23
    num_strikes = 289
    num_expiries = 53

    model = build_parameter_network(23, num_strikes * num_expiries)
    model.compile(optimizer='adam', loss={
        'mu': 'mse',
        'beta': 'mse', 
        'tau': 'mse',
        'w_volume': 'mse',
        'gamma_m': 'mse',
        'gamma_t': 'mse',
        'alpha_time': 'mse',
        'alpha_moneyness': 'mse'
        })
    
    trainset_dir = Path.cwd() / "nn_learning" / "training_data"
    






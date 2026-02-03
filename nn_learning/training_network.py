import tensorflow as tf
import numpy as np
from pathlib import Path

def build_parameter_network():
    # Input: (batch, T=23, num_contracts=289*53*2)
    num_contracts = 289 * 53 * 2  # = 30,634 contracts!
    inputs = tf.keras.layers.Input(shape=(23, num_contracts))
    
    # LSTM for temporal (only 23 steps, so lighter)
    x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(64)(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    
    # Output heads (same targets)
    outputs = {
        "mu": tf.keras.layers.Dense(1, name="mu")(x),
        "alpha_moneyness": tf.keras.layers.Dense(1, name="alpha_moneyness")(x),
        "alpha_time": tf.keras.layers.Dense(1, name="alpha_time")(x),
        "beta": tf.keras.layers.Dense(1, name="beta")(x),
        "gamma_t": tf.keras.layers.Dense(1, name="gamma_t")(x),
        "gamma_m": tf.keras.layers.Dense(1, name="gamma_m")(x),
        "w_volume": tf.keras.layers.Dense(1, name="w_volume")(x),
        "rho_self": tf.keras.layers.Dense(1, name="rho_self")(x),
        "tau": tf.keras.layers.Dense(4, activation="sigmoid", name="tau")(x)
    }
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

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
    
    # model.fit uses dimensions: (batch_size, T, num_contracts)
    
    trainset_dir = Path.cwd() / "nn_learning" / "training_data"
    






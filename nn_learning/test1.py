import tensorflow as tf

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

def hawkes_log_likelihood(y_true, y_pred, events, T):
    """
    y_true: Observed data (not used directly, but needed for Keras).
    y_pred: Predicted parameters (mu, alpha_moneyness, alpha_time, beta, etc.).
    events: shape (batch_size, T, num_expiries, num_strikes, num_types) representing event counts.
    T: Total time steps.
    """
    # Unpack predicted parameters
    mu = y_pred["mu"]
    alpha_moneyness = y_pred["alpha_moneyness"]
    alpha_time = y_pred["alpha_time"]
    beta = y_pred["beta"]
    gamma_t = y_pred["gamma_t"]
    gamma_m = y_pred["gamma_m"]
    tau = tf.reshape(y_pred["tau"], (-1, num_types, num_types))
    w_volume = y_pred["w_volume"]

    # Compute the intensity lambda*(t) for each time step
    # This requires simulating the Hawkes process with the predicted parameters
    # You can use TensorFlow to implement the intensity calculation
    lambda_star = compute_intensity(
        mu, alpha_moneyness, alpha_time, beta, gamma_t, gamma_m, tau, w_volume, events, T
    )

    # Compute the log-likelihood
    log_lik = 0.0
    for t in range(T):
        for exp in range(num_expiries):
            for strike in range(num_strikes):
                for typ in range(num_types):
                    n_events = events[:, t, exp, strike, typ]
                    log_lik += tf.reduce_sum(
                        n_events * tf.math.log(lambda_star[:, t, exp, strike, typ] + 1e-10) -
                        lambda_star[:, t, exp, strike, typ] * dt
                    )

    return -log_lik  # Negative log-likelihood as loss

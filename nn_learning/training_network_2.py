import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import sys
from nn_learning.loss_funcs import *
tf.debugging.enable_check_numerics()

def iterprod(*args):

    n = len(args)
    iter_idx = np.ones(n, dtype=int)
    prod = np.prod(args)
    count = 1
    while count < prod:
        yield count, iter_idx
        count += 1
        for k in range(-1, -n-1, -1):
            if iter_idx[k] == args[k]:
                continue
            iter_idx[k] += 1
            if k != -1:
                iter_idx[k + 1:] = 1
            break

    yield count, iter_idx

def pad_contracts(input_struct, C_max):
    """
    input_struct: (T, C_subset, F)
    returns: (T, C_max, F)
    """
    T, C_subset, F = input_struct.shape

    padded = np.zeros((T, C_max, F), dtype=input_struct.dtype)

    padded[:, :C_subset, :] = input_struct

    # mask: 1 = real contract, 0 = padding
    mask = np.zeros((T, C_max, 1))
    mask[:, :C_subset, :] = 1.0

    return padded, mask

def random_contract_subset(input_struct, max_contracts):
    _, T, N, _ = input_struct.shape
    
    idx = np.random.choice(N, size=max_contracts, replace=False)
    return input_struct[:, :, idx, :]

def build_parameter_network(T, num_contracts):
    # Main input for volumes, deltas, gammas

    inputs = tf.keras.layers.Input(shape=(T, num_contracts, 5), name="input") # volume, delta, gamma, moneyness, normalized_time_till_expiry
    mask_input = tf.keras.layers.Input(shape=(T, num_contracts, 1), name="mask")

    """
    # Reshape main input features
    x = tf.keras.layers.Reshape((T, num_contracts, 5))(inputs)  # Shape: (batch_size, T, num_contracts, 5)

    # Reshape for LSTM layers
    x = tf.keras.layers.Reshape((T, num_contracts * 5))(x)  # Shape: (batch_size, T, num_contracts * 19)

    # LSTM layers
    x = tf.keras.layers.LSTM(32, return_sequences=True)(x)  # Shape: (batch_size, T, 32)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(32)(x)  # Shape: (batch_size, 32)

    # Dense layers
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    """
    # timedistributed layer maps each (contract, timestamp) to a higher dimensional
    # embedding to learn more about the underlying structure
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(32, activation="relu")
    )(inputs)

    x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(32, activation="relu")
        )(x)

    # aggregation over contracts: learn general contract space layout, not
    # individual contracts themselves
    #x = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=2))(x)
    
    # attn = tf.keras.layers.Dense(1, activation="tanh")(x)
    # weights = tf.keras.layers.Softmax(axis=2)(attn)

    # x = tf.keras.layers.Multiply()([x, weights])
    # x = tf.keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=2))(x)

    attn = tf.keras.layers.Dense(1, activation="tanh")(x)

    # apply mask: set padded logits to very negative
    attn = tf.keras.layers.Add()([
        attn,
        (1.0 - mask_input) * (-1e9)
    ])

    weights = tf.keras.layers.Softmax(axis=2)(attn)

    x = tf.keras.layers.Multiply()([x, weights])
    x = tf.keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=2))(x)

    # shape: (batch, T, 32)

    # temporal learning
    x = tf.keras.layers.LSTM(32, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(32)(x)

    # Outputs
    outputs = {
        "mu_intensity": tf.keras.layers.Dense(1, name="mu_intensity")(x),
        "alpha_moneyness": tf.keras.layers.Dense(1, name="alpha_moneyness")(x),
        "alpha_time": tf.keras.layers.Dense(1, name="alpha_time")(x),
        "beta": tf.keras.layers.Dense(1, name="beta")(x),
        "gamma_t": tf.keras.layers.Dense(1, name="gamma_t")(x),
        "gamma_m": tf.keras.layers.Dense(1, name="gamma_m")(x),
        "volume_base": tf.keras.layers.Dense(1, activation="softplus", name="volume_base")(x),
        "w_volume": tf.keras.layers.Dense(1, name="w_volume")(x),
        "rho_self": tf.keras.layers.Dense(1, activation="sigmoid", name="rho_self")(x),
        "tau": tf.keras.layers.Dense(4, activation="sigmoid", name="tau")(x),
        "buy_base_param": tf.keras.layers.Dense(1, name="buy_base_param")(x),
        "buy_imbalance_param": tf.keras.layers.Dense(1, name="buy_imbalance_param")(x),
        "limit_base_param": tf.keras.layers.Dense(1, name="limit_base_param")(x),
        "limit_vol_param": tf.keras.layers.Dense(1, name="limit_vol_param")(x),
        "limit_distance_param": tf.keras.layers.Dense(1, name="limit_distance_param")(x),
        "limit_spread_param": tf.keras.layers.Dense(1, name="limit_spread_param")(x),
        "kappa": tf.keras.layers.Dense(1, name="kappa")(x),
        "theta": tf.keras.layers.Dense(1, name="theta")(x),
        "xi": tf.keras.layers.Dense(1, name="xi")(x),
        "mu": tf.keras.layers.Dense(1, name="mu")(x),
        "rho": tf.keras.layers.Dense(1, name="rho")(x),
    }

    #model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model = tf.keras.Model(
        inputs=[inputs, mask_input],
        outputs=outputs
    )

    model.compile(
        optimizer='adam',
        loss={
            "mu_intensity": custom_objects['weighted_mse_2'],
            "alpha_moneyness": custom_objects['relative_mse'],
            "alpha_time": custom_objects['relative_mse'],
            "beta": custom_objects['weighted_mse_3'],
            "gamma_t": custom_objects['relative_mse'],
            "gamma_m": custom_objects['relative_mse'],
            "volume_base": custom_objects['log_mse'],
            "w_volume": custom_objects['relative_mse'],
            "rho_self": custom_objects['bounded_mse'],
            "tau": custom_objects['bounded_mse'],
            "buy_base_param": custom_objects['bounded_mse'],
            "buy_imbalance_param": custom_objects['bounded_mse'],
            "limit_base_param": custom_objects['bounded_mse'],
            "limit_vol_param": custom_objects['bounded_mse'],
            "limit_spread_param": custom_objects['bounded_mse'],
            "limit_distance_param": custom_objects['bounded_mse'],
            "kappa": custom_objects['bounded_mse'],
            "theta": custom_objects['bounded_mse'],
            "xi": custom_objects['bounded_mse'],
            "mu": custom_objects['bounded_mse'],
            "rho": custom_objects['bounded_mse'],
        },
        metrics={
            "mu": "mae",
            "alpha_moneyness": "mae",
            "beta": "mae",
            "tau": "mae"
        }
    )

    return model

def convert_to_float(obj):
    if isinstance(obj, dict):
        return {k: convert_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_float(v) for v in obj]
    elif isinstance(obj, (int, float, str)) and not isinstance(obj, bool):
        try:
            return float(obj)
        except ValueError:
            return obj  # or raise an error if you want to be strict
    else:
        return obj


def prepare_input_data(dir):

    assetdata = np.load(dir / "assetdata.npy", allow_pickle=True) # shape (2, T)
    spot_prices = assetdata[0, :] # (T)
    
    spot_prices = spot_prices[:23]

    param_target = json.loads(json_data[dir.name])

    param_target = convert_to_float(param_target)

    expiry_dates = param_target["expiry_dts"]
    strike_prices = param_target["strike_prices"]

    last_expiry = max(expiry_dates) # in seconds after open

    normalized_expiries = np.empty((len(expiry_dates), T))
    moneynesses = np.empty((len(strike_prices), T))

    for t in range(T):
        delta_t = t * param_target["dt"] # timestep
        normalized_expiries[:, t] = [(expiry - delta_t) / last_expiry for expiry in expiry_dates]
        moneynesses[:, t] = [(strike - spot) / spot for strike, spot in zip(strike_prices, spot_prices)]

    param_target_flat = np.array([
        param_target["mu_intensity"],
        param_target["alpha_moneyness"],
        param_target["alpha_time"],
        param_target["beta"],
        param_target["gamma_t"],
        param_target["gamma_m"],
        param_target["volume_base"],
        param_target["w_volume"],
        param_target["rho_self"],
        *param_target["tau"][0],
        *param_target["tau"][1],
        param_target["buy_order_base_param"],
        param_target["buy_order_imbalance_param"],
        param_target["limit_order_base_param"],
        param_target["limit_order_vol_param"],
        param_target["limit_order_spread_param"],
        param_target["limit_order_distance_param"],
        param_target["kappa"],
        param_target["theta"],
        param_target["mu"],
        param_target["xi"],
        param_target["rho"],

    ])

    y_true_dict = {
        "mu_intensity": np.expand_dims(param_target_flat[0], axis=0).reshape(1, 1),
        "alpha_moneyness": np.expand_dims(param_target_flat[1], axis=0).reshape(1, 1),
        "alpha_time": np.expand_dims(param_target_flat[2], axis=0).reshape(1, 1),
        "beta": np.expand_dims(param_target_flat[3], axis=0).reshape(1, 1),
        "gamma_t": np.expand_dims(param_target_flat[4], axis=0).reshape(1, 1),
        "gamma_m": np.expand_dims(param_target_flat[5], axis=0).reshape(1, 1),
        "volume_base": np.expand_dims(param_target_flat[6], axis=0).reshape(1, 1),
        "w_volume": np.expand_dims(param_target_flat[7], axis=0).reshape(1, 1),
        "rho_self": np.expand_dims(param_target_flat[8], axis=0).reshape(1, 1),
        "tau": np.expand_dims(param_target_flat[9:13], axis=0).reshape(1, 4),
        "buy_base_param": np.expand_dims(param_target_flat[13], axis=0).reshape(1, 1),
        "buy_imbalance_param": np.expand_dims(param_target_flat[14], axis=0).reshape(1, 1),
        "limit_base_param": np.expand_dims(param_target_flat[15], axis=0).reshape(1, 1),
        "limit_vol_param": np.expand_dims(param_target_flat[16], axis=0).reshape(1, 1),
        "limit_distance_param": np.expand_dims(param_target_flat[17], axis=0).reshape(1, 1),
        "limit_spread_param": np.expand_dims(param_target_flat[18], axis=0).reshape(1, 1),
        "kappa": np.expand_dims(param_target_flat[19], axis=0).reshape(1, 1),
        "theta": np.expand_dims(param_target_flat[20], axis=0).reshape(1, 1),
        "mu": np.expand_dims(param_target_flat[21], axis=0).reshape(1, 1),
        "xi": np.expand_dims(param_target_flat[22], axis=0).reshape(1, 1),
        "rho": np.expand_dims(param_target_flat[23], axis=0).reshape(1, 1),
    }


    volume = np.load(dir / "traded_volumes.npy", allow_pickle=True) # shape (M, N, 2, T)
    volume = volume[:, :, :, :23] # cut off
    volume = np.reshape(volume, shape=(2, num_expiries * num_strikes, T)).T # shape (T, 2, M*N)

    overviews = np.load(dir / "overviews.npy", allow_pickle=True) # shape (M, N, 2, T)
    overviews = overviews[:, :, :, :23]

    deltas = np.empty((2, num_expiries * num_strikes, T)) # shape (2, M * N, T)
    gammas = np.empty((2, num_expiries * num_strikes, T)) # shape (2, M * N, T)

    deltas = deltas[:, :23]
    gammas = gammas[:, :23]

    # convert (N/M, T) structure to (2, M * N, T) structure (for expiries and strikes)
    expiries = np.empty((2, num_expiries * num_strikes, T))
    strikes = np.empty((2, num_expiries * num_strikes, T))

    # first iterate k, then m, then n
    # CALLS
    for idx, (n, m) in iterprod(num_strikes, num_expiries):
        overv = overviews[m - 1, n - 1, 0] # (T,)

        deltas[0, idx - 1] = [overv[t]["delta"] if overv[t] is not None else 0.0 for t in range(T)]
        gammas[0, idx - 1] = [overv[t]["gamma"] if overv[t] is not None else 0.0 for t in range(T)]

        expiries[0, idx - 1] = normalized_expiries[m - 1]
        strikes[0, idx - 1] = moneynesses[n - 1]

    # first iterate k, then m, then n
    # PUTS
    for idx, (n, m) in iterprod(num_strikes, num_expiries):
        overv = overviews[m - 1, n - 1, 1] # (T,)

        deltas[1, idx - 1] = [overv[t]["delta"] if overv[t] is not None else 0.0 for t in range(T)]
        gammas[1, idx - 1] = [overv[t]["gamma"] if overv[t] is not None else 0.0 for t in range(T)]

        expiries[1, idx - 1] = normalized_expiries[m - 1]
        strikes[1, idx - 1] = moneynesses[n - 1]

    volume = np.moveaxis(volume, 0, 2)
    volume = np.moveaxis(volume, 0, 1)
    input_struct = np.array([volume.T, deltas.T, gammas.T, expiries.T, strikes.T])

    # final shape: (5, T, num_contracts, 2) # 2: calls/puts

    return input_struct, y_true_dict

if __name__ == "__main__":

    """
    for spy_eod_202303.txt:
    counts: 6642
    strike count: 289
    exp counts: 53
    T: 23

    simplified model:
    contracts: 1000
    strikes: 10
    expiries: 10
    T: 23
    """

    # loss metrics
    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    val_loss_metric = tf.keras.metrics.Mean(name='val_loss')

    num_epochs = 10

    T = 23
    num_strikes = 4
    num_expiries = 4

    C_max = 64
    model = build_parameter_network(T, C_max)

    # model.fit uses dimensions: (batch_size, T, num_contracts)
    
    trainset_dir = Path.cwd() / "nn_learning" / "training_data"
    with open(trainset_dir / "param_info.json", "r") as f:
        json_data = json.load(f)

    #print('loaded paramset')

    loss_hist = {"loss": [], "val_loss": []} # each entry is list of lists

    train_pct = 0.9

    for epoch in range(num_epochs):

        train_loss_metric.reset_state()
        val_loss_metric.reset_state()
        epoch_losses = [] # store losses per epoch
        epoch_val_losses = [] # store validation losses per epoch

        dirs = [d for d in trainset_dir.iterdir() if d.name != "param_info.json"]
        np.random.shuffle(dirs)
        split_idx = int(train_pct * len(dirs))
        dirs_train = dirs[:split_idx]
        dirs_validate = dirs[split_idx:]

        for dir in dirs_train:
            
            input_struct, y_true_dict = prepare_input_data(dir)
            subset = random_contract_subset(input_struct, max_contracts=20)  # (T, 20, F)
            padded, mask = pad_contracts(subset, C_max=64)  # (T, 64, F)

            input_struct = np.expand_dims(padded, axis=0)   # (1, T, 64, F)
            mask = np.expand_dims(mask, axis=0)             # (1, T, 64, 1)

            #batch_loss = model.train_on_batch(input_struct, y_true_dict)
            batch_loss = model.train_on_batch(
                [input_struct, mask],
                y_true_dict
            )
            epoch_losses.append(batch_loss)

        for dir in dirs_validate:

            input_struct, y_true_dict = prepare_input_data(dir)
            
            subset = random_contract_subset(input_struct, max_contracts=20)  # (T, 20, F)
            padded, mask = pad_contracts(subset, C_max=64)  # (T, 64, F)

            input_struct = np.expand_dims(padded, axis=0)   # (1, T, 64, F)
            mask = np.expand_dims(mask, axis=0)             # (1, T, 64, 1)

            #batch_loss = model.test_on_batch(input_struct, y_true_dict)
            batch_loss = model.train_on_batch(
                [input_struct, mask],
                y_true_dict
            )       
            epoch_val_losses.append(batch_loss)

        print('appending epoch losses:', epoch_losses)
        print('appending epoch val losses:', epoch_val_losses)

        loss_hist["loss"].append(epoch_losses)
        loss_hist["val_loss"].append(epoch_val_losses)

    model.save(Path.cwd() / "nn_learning" / "testmodel.keras")

    print('lost hist:'); print(loss_hist)

    with open(Path.cwd() / "nn_learning" / "loss_history", "w") as f:
        json.dump(loss_hist, f)


"""
python nn_learning/training_network.py
"""
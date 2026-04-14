import tensorflow as tf
import numpy as np
from pathlib import Path
import json
from keras import backend as K
import pandas as pd
import sys
from nn_learning.loss_funcs import *
from keras.layers import Dense, TimeDistributed, Lambda, Multiply, Dropout, LSTM, Softmax
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

# NOT USED
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



def build_parameter_network(T, num_contracts):
    # Main input for volumes, deltas, gammas

    inputs = tf.keras.layers.Input(shape=(T, num_contracts, 6), name="input") # volume, delta, gamma, moneyness, normalized_time_till_expiry
    
    x = TimeDistributed(Dense(32, activation="relu"))(inputs)
    x = TimeDistributed(Dense(32, activation="relu"))(x)

    attn = Dense(1)(x)
    weights = Softmax(axis=2)(attn)

    x = Multiply()([x, weights])

    x = Lambda(sum_over_contracts)(x)  # (batch, T, 32)

    x = LSTM(32, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)

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
        inputs=inputs,
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

first_quote = pd.to_datetime("2023-03-01 16:00:00") # treat as starting date
last_quote = pd.to_datetime("2023-03-31 16:00:00") # treat as last date
ref_diff = (last_quote - first_quote).total_seconds()

T = 30
num_strikes = 10
num_expiries = 10

num_contracts = 2 * num_expiries * num_strikes

def prepare_input_data(dir):

    assetdata = np.load(dir / "assetdata.npy", allow_pickle=True) # shape (2, T)
    spot_prices = assetdata[0, :] # (T)

    param_target = json_data[dir.name]
    actual_params = json.loads(param_target["params"])

    param_target = convert_to_float(param_target)

    expiry_dates = pd.to_datetime(param_target["expiries"])
    strike_prices = convert_to_float(param_target["strikes"])

    normalized_expiries = np.empty((len(expiry_dates), T))
    moneynesses = np.empty((len(strike_prices), T))

    for t in range(T):
        delta_t = t * actual_params["dt"] # timestep
        #normalized_expiries[:, t] = [(expiry - first_quote).total_seconds() / ref_diff for expiry in expiry_dates]
        normalized_expiries[:, t] = [delta_t / (expiry - first_quote).total_seconds() for expiry in expiry_dates]
        moneynesses[:, t] = [(strike - spot) / spot for strike, spot in zip(strike_prices, spot_prices)]


    param_target_flat = np.array([
        actual_params["mu_intensity"],
        actual_params["alpha_moneyness"],
        actual_params["alpha_time"],
        actual_params["beta"],
        actual_params["gamma_t"],
        actual_params["gamma_m"],
        actual_params["volume_base"],
        actual_params["w_volume"],
        actual_params["rho_self"],
        *actual_params["tau"][0],
        *actual_params["tau"][1],
        actual_params["buy_order_base_param"],
        actual_params["buy_order_imbalance_param"],
        actual_params["limit_order_base_param"],
        actual_params["limit_order_vol_param"],
        actual_params["limit_order_spread_param"],
        actual_params["limit_order_distance_param"],
        actual_params["kappa"],
        actual_params["theta"],
        actual_params["mu"],
        actual_params["xi"],
        actual_params["rho"],

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
    volume = np.reshape(volume, shape=(2, num_expiries * num_strikes, T)) # shape (T, 2, M*N) ;;

    overviews = np.load(dir / "overviews.npy", allow_pickle=True) # shape (M, N, 2, T)

    idx_global = 0

    volume_all   = np.zeros((num_contracts, T))
    delta_all    = np.zeros((num_contracts, T))
    gamma_all    = np.zeros((num_contracts, T))
    expiry_all   = np.zeros((num_contracts, T))
    moneyness_all= np.zeros((num_contracts, T))
    cp_flag_all  = np.zeros((num_contracts, T))

    for cp in range(2):  # 0 = call, 1 = put
        cp_flag = 1.0 if cp == 0 else -1.0

        for idx, (n, m) in iterprod(num_strikes, num_expiries):
            overv = overviews[m - 1, n - 1, cp]

            for t in range(T):
                if overv[t] is not None:
                    delta_all[idx_global, t] = overv[t]["delta"]
                    gamma_all[idx_global, t] = overv[t]["gamma"]
                else:
                    delta_all[idx_global, t] = 0.0
                    gamma_all[idx_global, t] = 0.0

            expiry_all[idx_global]    = normalized_expiries[m - 1]
            moneyness_all[idx_global] = moneynesses[n - 1]
            volume_all[idx_global]    = volume[cp, idx - 1]

            cp_flag_all[idx_global]   = cp_flag  # ✅ THIS IS THE KEY

            idx_global += 1

    input_struct = np.stack([
        volume_all.T,
        delta_all.T,
        gamma_all.T,
        moneyness_all.T,
        expiry_all.T,
        cp_flag_all.T
    ], axis=-1)

    # input_struct shape: (30, 200, 6)

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

    num_contracts = 2 * num_strikes * num_expiries
    model = build_parameter_network(T, num_contracts)

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
            # input struct shape: (30, 200, 6)
            input_struct = np.expand_dims(input_struct, axis=0)
            batch_loss = model.train_on_batch(input_struct, y_true_dict)
            batch_loss = [float(nparray) for nparray in batch_loss]
            epoch_losses.append(batch_loss)

        for dir in dirs_validate:

            input_struct, y_true_dict = prepare_input_data(dir)
            input_struct = np.expand_dims(input_struct, axis=0)
            batch_loss = model.test_on_batch(input_struct, y_true_dict)
            batch_loss = [float(nparray) for nparray in batch_loss]
            epoch_val_losses.append(batch_loss)

        #print('appending epoch losses:', epoch_losses)
        #print('appending epoch val losses:', epoch_val_losses)
        print('lengths:', len(epoch_val_losses), len(epoch_val_losses[0]), len(epoch_val_losses[1]))
        print('lengths:', len(epoch_losses), len(epoch_losses[0]), len(epoch_losses[1]))

        loss_hist["loss"].append(epoch_losses)
        loss_hist["val_loss"].append(epoch_val_losses)

    model.save(Path.cwd() / "nn_learning" / "testmodel.keras")

    print('lost hist:'); print(loss_hist)

    with open(Path.cwd() / "nn_learning" / "loss_history.json", "w") as f:
        json.dump(loss_hist, f)
    print('done')

"""
python nn_learning/training_network_2.py
"""
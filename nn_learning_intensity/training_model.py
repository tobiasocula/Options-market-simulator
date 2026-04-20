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

def build_model(T, num_contracts):

    inputs = tf.keras.Input(shape=(T, num_contracts, 4))

    # --- CONTRACT ENCODING ---
    x = TimeDistributed(Dense(32, activation="relu"))(inputs)
    x = TimeDistributed(Dense(32, activation="relu"))(x)

    # --- CROSS-CONTRACT ATTENTION ---
    attn = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=32)
    print("x shape before attention:", x.shape)
    x = attn(x, x)
    print("x shape after attention:", x.shape)

    # reduce contracts AFTER interaction
    x = tf.keras.layers.Lambda(sum_over_contracts)(x)

    # --- TIME MODEL ---
    x = LSTM(32)(x)

    # --- MULTI-HEAD OUTPUT ---
    mu = Dense(1, name="mu")(x)
    alpha_m = Dense(1, name="alpha_moneyness")(x)
    alpha_t = Dense(1, name="alpha_time")(x)
    beta = Dense(1, name="beta")(x) 

    model = tf.keras.Model(inputs, [mu, alpha_m, alpha_t, beta])
    model.compile(
        optimizer='adam',
        loss={
            "mu": "mse",
            "alpha_moneyness": "mse",
            "alpha_time": "mse",
            "beta": "mse"
        },
        metrics={
            "mu": "mae",
            "alpha_moneyness": "mae",
            "alpha_time": "mae",
            "beta": "mae"
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


def needed_data(dirs):
    # 4 = num_features: [volume, moneyness, expiry, cp_flag]
    
    norm = np.empty((len(dirs), T, num_contracts, 4))
    logmus = []

    for i, dir in enumerate(dirs):

        param_target = json_data[dir.name]
        expiry_dates = pd.to_datetime(param_target["expiries"])
        strike_prices = convert_to_float(param_target["strikes"])
        logmu = np.log(json.loads(param_target["params"])["mu_intensity"])

        volume = np.load(dir / "traded_volumes.npy", allow_pickle=True) # shape (M, N, 2, T)
        volume_norm = (volume - np.mean(volume)) / (1e-8 + np.std(volume))

        strikes_norm = (strike_prices - np.mean(strike_prices)) / (1e-8 + np.std(strike_prices))
        expiries_norm = [(exp - first_quote).total_seconds() / (last_quote - first_quote).total_seconds()
                         for exp in expiry_dates]
        
        idx_global = 0
        volume_all   = np.zeros((num_contracts, T))
        expiry_all   = np.zeros((num_contracts, T))
        moneyness_all= np.zeros((num_contracts, T))
        cp_flag_all  = np.zeros((num_contracts, T))
        
        for cp in range(2):  # 0 = call, 1 = put
            cp_flag = 1.0 if cp == 0 else -1.0

            for _, (n, m) in iterprod(num_strikes, num_expiries):

                expiry_all[idx_global]    = expiries_norm[m - 1]
                moneyness_all[idx_global] = strikes_norm[n - 1]
                volume_all[idx_global]    = volume_norm[m - 1, n - 1, cp]
                cp_flag_all[idx_global]   = cp_flag

        input_struct = np.stack([
            volume_all.T,
            moneyness_all.T,
            expiry_all.T,
            cp_flag_all.T
        ], axis=-1)

        norm[i,:,:,:] = input_struct # (100, 200, 4)
        print('input struct shape:', input_struct.shape)
        logmus.append(logmu)

    print('norm shape:', norm.shape)

    return norm, logmus # latter: (num_dirs, T, num_contracts, features)    

num_strikes = 10
num_expiries = 10

# loss metrics
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
val_loss_metric = tf.keras.metrics.Mean(name='val_loss')

num_epochs = 10
T = 100

num_contracts = 2 * num_strikes * num_expiries
model = build_model(T, num_contracts)

trainset_dir = Path.cwd() / "nn_learning_intensity" / "training_data"
with open(trainset_dir / "param_info.json", "r") as f:
    json_data = json.load(f)

loss_hist = {"loss": [], "val_loss": []} # each entry is list of lists
train_pct = 0.9

dirs = [d for d in trainset_dir.iterdir() if d.name != "param_info.json"]
np.random.shuffle(dirs)
split_idx = int(train_pct * len(dirs))
dirs_train = dirs[:split_idx]
dirs_validate = dirs[split_idx:]

X_train, y_train = needed_data(dirs)

print('x train shape:', X_train.shape)

y_true = np.array([
    [np.log(json.loads(json_data[dir.name]["params"])["mu_intensity"]) for dir in dirs],
    [np.log(json.loads(json_data[dir.name]["params"])["alpha_moneyness"]) for dir in dirs],
    [np.log(json.loads(json_data[dir.name]["params"])["alpha_time"]) for dir in dirs]
]).T

y_train = {
    "mu": y_true[:, 0],
    "alpha_moneyness": y_true[:, 1],
    "alpha_time": y_true[:, 2]
}

history = model.fit(
    X_train,
    y_train,
    batch_size=1,
    epochs=50,
    validation_split=0.1,
    verbose=1
)


save_dir = Path.cwd() / "nn_learning_intensity" / "results"
save_dir.mkdir(exist_ok=True)

with open(save_dir / "training_history.json", "w") as f:
    json.dump(history.history, f)

with open(save_dir / "train_dirs.json", "w") as f:
    json.dump([d.name for d in dirs_train], f)

with open(save_dir / "val_dirs.json", "w") as f:
    json.dump([d.name for d in dirs_validate], f)

model.save(save_dir / "model.keras")

import matplotlib.pyplot as plt

plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.legend()
plt.title("Loss")
plt.show()

# import os
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
# os.environ["TF_DISABLE_XLA_JIT"] = "1"
import os

import tensorflow as tf
tf.config.optimizer.set_jit(False)
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import sys
# import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))

import tensorflow as tf

with tf.device('/GPU:0'):
    x = tf.random.normal((3000, 3000))
    y = tf.matmul(x, x)

print("Success:", y.shape)


import tensorflow as tf
import numpy as np
from pathlib import Path
import json
from keras import backend as K
import pandas as pd

from loss_funcs import *
from keras.layers import Dense, TimeDistributed, Lambda, Multiply, Dropout, LSTM, Softmax

"""

source tf_gpu_env/bin/activate

python nn_learning_intensity/training_model.py
"""

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

def reshape_for_attention(x):
    shape = tf.shape(x)
    batch = shape[0]
    return tf.reshape(x, (batch * T, num_contracts, 32))

def reshape_back(x):
    shape = tf.shape(x)
    batch = shape[0] // T
    return tf.reshape(x, (batch, T, num_contracts, 32))

def build_model(T, num_contracts):

    inputs = tf.keras.Input(shape=(T, num_contracts, 4))

    # --- CONTRACT ENCODING ---
    x = TimeDistributed(Dense(32, activation="relu"))(inputs)
    x = TimeDistributed(Dense(32, activation="relu"))(x)

    # # --- CROSS-CONTRACT ATTENTION ---
    attn = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=32)
    
    # # reshape for attention over contracts
    # x = tf.reshape(x, (-1, num_contracts, 32))
    # x = attn(x, x)
    # x = tf.reshape(x, (-1, T, num_contracts, 32))

    x = Lambda(reshape_for_attention)(x)
    x = attn(x, x)
    x = Lambda(reshape_back)(x)

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
        },
        run_eagerly=True # disables graph + XLA, avoids ALL JIT issues, more stable but slower runtime
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


def needed_data(dirs, dirs_params):
    # 4 = num_features: [volume, moneyness, expiry, cp_flag]
    
    norm = np.empty((len(dirs), T, num_contracts, 4))
    logmus = []
    ats = []
    ams = []
    betas = []

    for i, (dir, dir_param) in enumerate(zip(dirs, dirs_params)):

        with open(dir_param, "r") as f:
            json_data = json.load(f)

        expiry_dates = pd.to_datetime(json_data["expiries"])
        strike_prices = convert_to_float(json_data["strikes"])
        logmu = np.log(json.loads(json_data["params"])["mu_intensity"])
        beta = json.loads(json_data["params"])["beta"]
        alpha_t = json.loads(json_data["params"])["alpha_time"]
        alpha_m = json.loads(json_data["params"])["alpha_moneyness"]

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
        betas.append(beta)
        ams.append(alpha_m)
        ats.append(alpha_t)
        idx_global += 1

    return norm, np.array([
        logmus, ams, ats, betas
    ])


num_strikes = 10
num_expiries = 10

# loss metrics
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
val_loss_metric = tf.keras.metrics.Mean(name='val_loss')

num_epochs = 10
T = 100

num_contracts = 2 * num_strikes * num_expiries
model = build_model(T, num_contracts)

trainset_dir = Path.cwd() / "nn_learning_intensity" / "trainset"

loss_hist = {"loss": [], "val_loss": []} # each entry is list of lists
train_pct = 0.9

dirs = [d for d in trainset_dir.iterdir() if d.name[0] == "s"]
dirs_params = [d for d in trainset_dir.iterdir() if d.name[0] == "j"]

ints_dirs = [int(d.name.split('_')[-1]) for d in dirs]
ints_dirs_params = [int(d.name.split('_')[-1].split('.')[0]) for d in dirs_params]
all_ints = set(ints_dirs).intersection((set(ints_dirs_params)))

dirs = [d for d in dirs if int(d.name.split('_')[-1]) in all_ints]
dirs_params = [d for d in dirs_params if int(d.name.split('_')[-1].split('.')[0]) in all_ints]


assert len(dirs) == len(dirs_params), f"error: {len(dirs)} vs {len(dirs_params)}"
idx = np.random.permutation(len(dirs))
print('idx:'); print(idx)
dirs = np.array(dirs)
dirs_params = np.array(dirs_params)
dirs = dirs[idx]
dirs_params = dirs_params[idx]

split_idx = int(train_pct * len(dirs))
dirs_train = dirs[:split_idx]
dirs_validate = dirs[split_idx:]
dirs_params_train = dirs_params[:split_idx]
dirs_params_validate = dirs_params[split_idx:]

X_train, y_train = needed_data(dirs, dirs_params)
y_train = y_train.T

print('x train shape:', X_train.shape) # (184, 100, 200, 4)
print('y_train shape:', y_train.shape) # (4, 184)

y_train = {
    "mu": y_train[:,0],
    "alpha_moneyness": y_train[:,1],
    "alpha_time": y_train[:,2],
    "beta": y_train[:,3],
}

history = model.fit(
    X_train,
    y_train,
    batch_size=10,
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


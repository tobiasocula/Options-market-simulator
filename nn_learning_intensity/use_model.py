import numpy as np
from pyDOE import lhs
from pathlib import Path
from cross_excitation import cross_excitation
from param_class import CrossExcitation
from debug import Debugger
import json
import pandas as pd
import sys

dir = Path.cwd() / "nn_learning_intensity" / "training_data"
json_path = dir / "param_info.json"

with open(json_path, "r") as f:
    json_data = json.load(f)

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

T = 100
num_expiries = 10
num_strikes = 10
num_contracts = 2*num_expiries*num_strikes

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
        normalized_expiries[:, t] = [delta_t / (expiry - first_quote).total_seconds() for expiry in expiry_dates]
        moneynesses[:, t] = [(strike - spot) / spot for strike, spot in zip(strike_prices, spot_prices)]

    param_target_flat = actual_params["mu_intensity"]

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

            cp_flag_all[idx_global]   = cp_flag

            idx_global += 1

    input_struct = np.stack([
        volume_all.T,
        delta_all.T,
        gamma_all.T,
        moneyness_all.T,
        expiry_all.T,
        cp_flag_all.T
    ], axis=-1)

    y_true = np.array([[np.log(actual_params["mu_intensity"])]])

    return input_struct, y_true

import tensorflow as tf
from nn_learning.loss_funcs import custom_objects
model_path = Path.cwd() / "nn_learning_intensity" / "results" / "mu_model.keras"

model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

for d in dir.iterdir():
    if d.name == "param_info.json":
        continue
    input_struct, y_true = prepare_input_data(d)
    input_struct = np.expand_dims(input_struct, axis=0)  # shape (1, 100, 200, 6)
    results = model.predict(input_struct)
    print("predicted:", results, "versus real:", y_true)
    
    
    

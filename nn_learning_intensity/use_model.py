import numpy as np
from pyDOE import lhs
from pathlib import Path
import json
import pandas as pd
import sys

trainset_dir = Path.cwd() / "nn_learning_intensity" / "trainset"


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

def prepare_input_data(dirs, dirs_params):

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
        logmus.append(logmu)
        betas.append(beta)
        ams.append(alpha_m)
        ats.append(alpha_t)
        idx_global += 1

    return norm, np.array([
        logmus, ams, ats, betas
    ])

import tensorflow as tf
from loss_funcs import *

model_path = Path.cwd() / "nn_learning_intensity" / "results" / "model.keras"

model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

dirs = [d for d in trainset_dir.iterdir() if d.name[0] == "s"]
dirs_params = [d for d in trainset_dir.iterdir() if d.name[0] == "j"]

ints_dirs = [int(d.name.split('_')[-1]) for d in dirs]
ints_dirs_params = [int(d.name.split('_')[-1].split('.')[0]) for d in dirs_params]
all_ints = set(ints_dirs).intersection((set(ints_dirs_params)))

dirs = [d for d in dirs if int(d.name.split('_')[-1]) in all_ints]
dirs_params = [d for d in dirs_params if int(d.name.split('_')[-1].split('.')[0]) in all_ints]

x, y = prepare_input_data(dirs, dirs_params)
print(y.shape) # (4, 184)
print(x.shape) # (184, 100, 200, 4)

results = model.predict(x)
print(len(results))


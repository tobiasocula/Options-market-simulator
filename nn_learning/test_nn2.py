#import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import pandas as pd
import tensorflow as tf
import itertools
import sys
from nn_learning.loss_funcs import custom_objects

import keras
keras.config.enable_unsafe_deserialization()

def iterprod(*args):

    n = len(args)
    iter_idx = np.ones(n)
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

df = pd.read_csv(Path.cwd() / "data" / "spy_eod_202303.txt", index_col=0)
df.columns = [k.strip(' ') for k in df.columns]

quotetimes = df.index.unique()

print('len quotetimes:', len(quotetimes)) # 23

model_path = Path.cwd() / "nn_learning" / "testmodel.keras"
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

groups = df.groupby(["[QUOTE_READTIME]"])

selected_strikes = [120.0, 130.0, 140.0, 150.0]
selected_expiries = [' 2023-03-06', ' 2023-03-13', ' 2023-04-13', ' 2023-03-22']

num_contracts = 2*len(selected_expiries)*len(selected_strikes)
T = 23

quote_counter = 0
contract_counter = 0

# structures for input nn
volume = np.zeros((T, num_contracts)) # in order: (n, m, k)
delta = np.zeros((T, num_contracts))
gamma = np.zeros((T, num_contracts))
strikes_struct = np.zeros((T, num_contracts))
expiries_struct = np.zeros((T, num_contracts))

for strike, expiry in itertools.product(selected_strikes, selected_expiries):
    
    wanted = df[(df["[STRIKE]"] == strike) & df["[EXPIRE_DATE]"] == expiry]

    print(len(wanted))
    continue
    print('break')
        # volume[quote_counter, contract_counter] = row["[C_VOLUME]"]
        # volume[t, contract_put] = row["[P_VOLUME]"]
        # delta[t, contract_call] = row.get("[C_DELTA]", 0.0)
        # delta[t, contract_put] = row.get("[P_DELTA]", 0.0)
        # gamma[t, contract_call] = row.get("[C_GAMMA]", 0.0)
        # gamma[t, contract_put] = row.get("[P_GAMMA]", 0.0)
        # strikes_struct[t, contract_call] = (strike - spot) / spot
        # strikes_struct[t, contract_put] = (strike - spot) / spot
        # expiries_struct[t, contract_call] = (expiry - quotetime) / max_expiry_unix
        # expiries_struct[t, contract_put] = (expiry - quotetime) / max_expiry_unix

        # print('volumes:', volume[t, contract_call], volume[t, contract_put])
        # print('deltas:', delta[t, contract_call], delta[t, contract_put])
        # print('gamma:', gamma[t, contract_call], gamma[t, contract_put])
        # print('strikes:', strikes_struct[t, contract_call], strikes_struct[t, contract_put])
        # print('expiries:', expiries_struct[t, contract_call], expiries_struct[t, contract_put])

sys.exit()

# desired shape: (T, num_contracts, 5)
input_struct = np.array([volume, delta, gamma, strikes_struct, expiries_struct]) # shape: (5, T, num_contracts)
input_struct = np.moveaxis(input_struct, 0, 1) # (T, 5, num_contracts)
input_struct = np.moveaxis(input_struct, 1, 2) # (T, num_contracts, 5)
input_struct = np.expand_dims(input_struct, axis=0)

result = model.predict(input_struct)
print('result:'); print(result)

def transform(dict):
    new = {}
    for k,v in dict.items():
        if k == "tau":
            new[k] = v.flatten().tolist()
        elif isinstance(v, np.ndarray):
            new[k] = float(v.flatten().flatten()[0])
        elif isinstance(v, np.float32):
            new[k] = float(v)
    return new

# save result
with open(Path.cwd() / "nn_learning" / "predicted_params.json", "w") as f:
    json.dump(transform(result), f)
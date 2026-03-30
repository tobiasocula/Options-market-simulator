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

"""

# Count occurrences of each expiry and strike
expiry_counts = df["[EXPIRE_DATE]"].value_counts()
strike_counts = df["[STRIKE]"].value_counts()
print('expiry counts:'); print(expiry_counts)

# Select top K expiries and top L strikes
selected_expiries = expiry_counts.nlargest(num_expiries).index
selected_strikes = strike_counts.nlargest(num_strikes).index

print('selected expiries:'); print(selected_expiries)
print('selected strikes:'); print(selected_strikes)


"""


"""
selected expiries:
Index([' 2023-06-16', ' 2023-03-31', ' 2023-04-21', ' 2023-05-19'], dtype='object', name='[EXPIRE_DATE]')
selected strikes:
Index([400.0, 410.0, 390.0, 380.0, 395.0, 405.0, 370.0], dtype='float64', name='[STRIKE]')

"""

selected_strikes = [120.0, 130.0, 140.0, 150.0]
selected_expiries = [' 2023-06-16', ' 2023-03-31', ' 2023-04-21', ' 2023-05-19']
selected_expiries = df[df["[EXPIRE_DATE]"].isin(selected_expiries)]["[EXPIRE_UNIX]"].unique()

assert len(selected_expiries) == 4, AssertionError()

max_expiry_unix = selected_expiries.max()

# # select only ones we need according to number of strikes and expiries
# selected_strikes = selected_strikes[:num_strikes]
# selected_expiries = selected_expiries[:num_expiries]

print('selected expiries:'); print(selected_expiries)


num_expiries = 4
num_strikes = 4

T = 23
num_contracts = 2 * num_expiries * num_strikes
print('num contracts:', num_contracts)

# structures for input nn
volume = np.zeros((T, num_contracts)) # in order: (n, m, k)
delta = np.zeros((T, num_contracts))
gamma = np.zeros((T, num_contracts))
strikes_struct = np.zeros((T, num_contracts))
expiries_struct = np.zeros((T, num_contracts))

contract_mapping = {}
contract_idx = 0
for strike, expiry in itertools.product(selected_strikes, selected_expiries):
    contract_mapping[(strike, expiry, 0)] = contract_idx
    contract_idx += 1
    contract_mapping[(strike, expiry, 1)] = contract_idx
    contract_idx += 1

for t, quotetime in enumerate(quotetimes): # iterate 
    rows = df.loc[df.index == quotetime, :]
    rows = rows[rows["[EXPIRE_UNIX]"].isin(selected_expiries)]
    rows = rows[rows["[STRIKE]"].isin(selected_strikes)]

    print('rows:'); print(rows)
    # length should be 28
    if len(rows) != 28:
        print(f"len is not 28!: {len(rows)}")
    else:
        print('len was 28')

    rows = rows.sort_values(by=["[EXPIRE_UNIX]", "[STRIKE]"])
    for _, row in rows.iterrows():
        expiry = row["[EXPIRE_UNIX]"]
        strike = row["[STRIKE]"]
        spot = row["[UNDERLYING_LAST]"]

        contract_call = contract_mapping[(strike, expiry, 0)]
        contract_put = contract_mapping[(strike, expiry, 1)]

        volume[t, contract_call] = row["[C_VOLUME]"]
        volume[t, contract_put] = row["[P_VOLUME]"]
        delta[t, contract_call] = row.get("[C_DELTA]", 0.0)
        delta[t, contract_put] = row.get("[P_DELTA]", 0.0)
        gamma[t, contract_call] = row.get("[C_GAMMA]", 0.0)
        gamma[t, contract_put] = row.get("[P_GAMMA]", 0.0)
        strikes_struct[t, contract_call] = (strike - spot) / spot
        strikes_struct[t, contract_put] = (strike - spot) / spot
        expiries_struct[t, contract_call] = (expiry - quotetime) / max_expiry_unix
        expiries_struct[t, contract_put] = (expiry - quotetime) / max_expiry_unix

        print('volumes:', volume[t, contract_call], volume[t, contract_put])
        print('deltas:', delta[t, contract_call], delta[t, contract_put])
        print('gamma:', gamma[t, contract_call], gamma[t, contract_put])
        print('strikes:', strikes_struct[t, contract_call], strikes_struct[t, contract_put])
        print('expiries:', expiries_struct[t, contract_call], expiries_struct[t, contract_put])

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
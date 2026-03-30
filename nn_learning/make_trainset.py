import numpy as np
from pyDOE import lhs
from pathlib import Path
from cross_excitation import cross_excitation
from param_class import CrossExcitation
from debug import Debugger
import json
import pandas as pd
import sys

# Define parameter ranges
param_ranges = {
    "mu_intensity": [1e-4, 1e-3],
    "alpha_moneyness": [0.010, 0.020],
    "alpha_time": [5e-6, 5e-5],
    "beta": [0.05, 0.5],
    "gamma_t": [5e-5, 5e-4],
    "gamma_m": [12.0, 20.0],
    "volume_base": [1.0, 2.0],
    "w_volume": [0.05, 0.1],
    "rho_self": [0.005, 0.01],
    "tau_self": [0.01, 0.01],
    "tau_cross": [0.005, 0.005],
    "kappa": [1.0, 3.0],
    "theta": [0.03, 0.09],
    "xi": [0.1, 0.4],
    "mu": [0.04, 0.08],
    "rho": [-0.9, -0.3],
    "limit_order_base_param": [0.4, 0.7],
    "limit_order_vol_param": [-1.5, -0.5],
    "limit_order_distance_param": [0.3, 0.8],
    "limit_order_spread_param": [0.3, 0.8],
    "buy_order_base_param": [-0.2, 0.2],
    "buy_order_imbalance_param": [1.0, 2.0]
}

dmode = 1
debugger = Debugger(mode=dmode)

json_path = Path.cwd() / "nn_learning" / "training_data" / "param_info.json"
if json_path.exists() and json_path.stat().st_size > 0:
    with open(json_path, "r") as f:
        json_data = json.load(f)
else:
    json_data = {}

# Number of samples
num_samples = 100

ranges = np.array(list(param_ranges.values()))

# Generate LHS samples (in [0, 1]!)
lhs_samples = lhs(len(param_ranges), samples=num_samples) # (num_samples, num_params)

# Scale each column to its range
for i, (low, high) in enumerate(param_ranges.values()):
    lhs_samples[:, i] = low + lhs_samples[:, i] * (high - low)

# print('mu intensities:')
# print(lhs_samples[0:10, 0])
# sys.exit()

choose_num_expiries = 10
choose_num_strikes = 10
using_strikes = np.load(Path.cwd() / "nn_learning" / "strikes_being_used.npy")
using_expiries = np.load(Path.cwd() / "nn_learning" / "expiries_being_used.npy")
using_expiries = pd.to_datetime(using_expiries)

"""
old strike sampling = too agressive
def sample_strikes():
    spot = 130 # this is the initial spot price, with which the simulation starts
    moneynesses = [((spot - strike) / spot)**2 for strike in using_strikes]
    # higher moneyness in abs value = lower chance of being picked
    # thus need to invert
    moneynesses = np.array([1 / (m + 1e-3) for m in moneynesses])
    moneynesses /= np.sum(moneynesses)

    return np.random.choice(using_strikes, size=choose_num_strikes, replace=False, p=moneynesses)
"""

def sample_strikes():
    spot = np.random.normal(130, 5)
    sigma = 1.0  # controls spread (tune this!)

    distances = (using_strikes - spot) / spot
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    weights /= weights.sum()

    return np.random.choice(using_strikes, size=choose_num_strikes, replace=False, p=weights)

first_quote = pd.to_datetime("2023-03-01 16:00:00") # treat as starting date
last_quote = pd.to_datetime("2023-03-31 16:00:00") # treat as last date

delta_t = (last_quote - first_quote).days # 30

def sample_expiries():
    tau = 10
    distances = (using_expiries - first_quote).days
    prob_dist = np.exp(-distances / tau)
    prob_dist /= np.sum(prob_dist) # sum should already be about one, but just to be sure
    return np.random.choice(using_expiries, size=choose_num_expiries, replace=False, p=prob_dist)

# Scale samples to the parameter ranges
for idx in range(lhs_samples.shape[0]):

    save = Path.cwd() / "nn_learning" / "training_data" / f"set_{idx}"
    save.mkdir(exist_ok=True)

    #chosen_strikes, chosen_expiries = random_subset()

    params = CrossExcitation(
    # TIME SCALE
    dt=100_000,
    T=30, # delete first two

    # STATIC BASE INTENSITY (per contract)
    alpha_moneyness = lhs_samples[idx][1],
    alpha_time      =  lhs_samples[idx][2],
    mu_intensity    = lhs_samples[idx][0],
    mu_variation = 0.1,

    # HAWKES DYNAMICS
    beta     = lhs_samples[idx][3],           # slower decay over ~1–2 days
    rho_self = lhs_samples[idx][8],          # self-excitation < 1
    tau      = [[lhs_samples[idx][9], lhs_samples[idx][10]],  # weaker cross-excitation
                [lhs_samples[idx][10], lhs_samples[idx][9]]],
    gamma_m  = lhs_samples[idx][5],            # smoother in strikes
    gamma_t  = lhs_samples[idx][4],            # smoother in expiries

    w_volume = lhs_samples[idx][7],           # lower volume impact

    # VOLUME MODEL (unchanged for now, tune after eyeballing data)
    contract_volume_mean = 2.0,
    contract_volume_std  = 0.5,
    volume_base          = lhs_samples[idx][6],
    volume_moneyness     = 0.5,
    volume_time_decay    = 0.002,

    # OPTION GRID
    #strike_prices = [120.0, 130.0, 140.0, 150.0],
    #expiry_dts    = [86400 * k for k in [5, 15, 30, 45]], # 5, 15, 30, 45 days resp.
    strike_prices = chosen_strikes,
    expiry_dts = chosen_expiries,

    # FINANCE MODEL (leave as is)
    risk_free     = 0.04,
    dividend_rate = 0.015,

    # INITIALIZATION (as you have; revisit later)
    base_n_orders_init       = 10,
    base_scale_init_orders   = 0.01,
    moneyness_scale_init_orders = 3.0,
    time_scale_init_orders   = 1.2,
    beta_init                = 0.1,
    gamma_init               = 0.5,

    init_open_price = 130.0,
    init_vola       = 0.04,

    # HESTON
    kappa = lhs_samples[idx][11],
    theta = lhs_samples[idx][12],
    xi    = lhs_samples[idx][13],
    mu    = lhs_samples[idx][14],
    rho   = lhs_samples[idx][15],

    # ORDER TYPE LOGIC (unchanged)
    limit_order_base_param    = lhs_samples[idx][16],
    limit_order_vol_param     = lhs_samples[idx][17],
    limit_order_distance_param= lhs_samples[idx][18],
    limit_order_spread_param  = lhs_samples[idx][19],

    buy_order_base_param      = lhs_samples[idx][20],
    buy_order_imbalance_param = lhs_samples[idx][21],
    )

    res = cross_excitation(params, save=True, savedir=save, debugger=debugger)
    if res is not None:
        # flag failed run
        continue

    json_data[f"set_{idx}"] = params.model_dump_json()

with open(json_path, "w") as f:
    json.dump(json_data, f)

"""
python nn_learning/make_trainset.py
"""

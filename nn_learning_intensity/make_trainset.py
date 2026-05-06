from pathlib import Path
import sys

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
from cross_excitation import cross_excitation
from param_class import CrossExcitation
from debug import Debugger
import json
import pandas as pd

param_ranges = {
    "mu_intensity": [5e-6, 5e-5],
    "alpha_moneyness": [0.010, 0.020],
    "alpha_time": [5e-6, 5e-5],
    "beta": [0.2, 0.7],
}

NUM_SAMPLES = 200
MU_SAMPLES = np.random.uniform(param_ranges["mu_intensity"][0], param_ranges["mu_intensity"][1], NUM_SAMPLES)
BETA_SAMPLES = np.random.uniform(param_ranges["beta"][0], param_ranges["beta"][1], NUM_SAMPLES)
ALPHA_M_SAMPLES = np.random.uniform(param_ranges["alpha_moneyness"][0], param_ranges["alpha_moneyness"][1], NUM_SAMPLES)
ALPHA_T_SAMPLES = np.random.uniform(param_ranges["alpha_time"][0], param_ranges["alpha_time"][1], NUM_SAMPLES)

dmode = 1
debugger = Debugger(mode=dmode)

choose_num_expiries = 10
choose_num_strikes = 10
using_strikes = np.load(Path.cwd() / "nn_learning_intensity" / "strikes_being_used.npy")
using_expiries = np.load(Path.cwd() / "nn_learning_intensity" / "expiries_being_used.npy")
using_expiries = pd.to_datetime(using_expiries)


first_expiry = using_expiries.min() - pd.Timedelta(days=2) # to exclude T = 0

def sample_strikes():
    spot = np.random.normal(130, 5)
    sigma = 0.2

    distances = (using_strikes - spot) / spot
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    weights /= weights.sum()

    return np.random.choice(using_strikes, size=choose_num_strikes, replace=False, p=weights)

first_quote = pd.to_datetime("2023-03-01 16:00:00") # treat as starting date
last_quote = pd.to_datetime("2023-03-31 16:00:00") # treat as last date

delta_t = (last_quote - first_quote).days # 30

def sample_expiries():
    tau = 50
    distances = (using_expiries - first_quote).days
    prob_dist = np.exp(-distances / tau)
    prob_dist /= np.sum(prob_dist) # sum should already be about one, but just to be sure
    return np.random.choice(using_expiries, size=choose_num_expiries, replace=False, p=prob_dist)

this_expiries_dts = sample_expiries() # generates pd.datetime values    
this_expiries = [(x - first_expiry).days*3600*24 for x in this_expiries_dts]
this_strikes = sample_strikes()



# Scale samples to the parameter ranges
for idx, (mu, beta, alphat, alpham) in enumerate(zip(MU_SAMPLES, BETA_SAMPLES, ALPHA_T_SAMPLES, ALPHA_M_SAMPLES), 122):

    save = Path.cwd() / "nn_learning_intensity" / "trainset" / f"set_{idx}"
    save.mkdir(exist_ok=True)

    params = CrossExcitation(
    # TIME SCALE
    dt=100_000,
    T=100,

    # STATIC BASE INTENSITY (per contract)
    alpha_moneyness = alpham,
    alpha_time      =   alphat,
    mu_intensity    = mu,
    mu_variation = 0.1,

    # HAWKES DYNAMICS
    beta     = beta,
    rho_self = 0.09,          # self-excitation < 1
    tau      = [[0.01, 0.005], [0.005, 0.01]],
    gamma_m  = 15.0,            # smoother in strikes
    gamma_t  = 1e-04,            # smoother in expiries

    w_volume = 0.08,           # lower volume impact

    # VOLUME MODEL (unchanged for now, tune after eyeballing data)
    contract_volume_mean = 2.0,
    contract_volume_std  = 0.5,
    volume_base          = 1.5,
    volume_moneyness     = 0.5,
    volume_time_decay    = 0.002,

    # OPTION GRID
    #strike_prices = [120.0, 130.0, 140.0, 150.0],
    #expiry_dts    = [86400 * k for k in [5, 15, 30, 45]], # 5, 15, 30, 45 days resp.
    strike_prices = this_strikes,
    expiry_dts = this_expiries,

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
    kappa = 2.0,
    theta = 0.05,
    xi    = 0.3,
    mu    = 0.06,
    rho   = -0.5,

    # ORDER TYPE LOGIC (unchanged)
    limit_order_base_param    = 0.5,
    limit_order_vol_param     = -1.0,
    limit_order_distance_param= 0.5,
    limit_order_spread_param  = 0.5,

    buy_order_base_param      = 0.0,
    buy_order_imbalance_param = 1.5,
    )

    res = cross_excitation(params, save=True, savedir=save, debugger=debugger)
    #res = cross_excitation(params, save=False, debugger=debugger)
    if res is not None:
        # flag failed run
        continue

    json_data = {
        "params": params.model_dump_json(),
        "expiries": [str(x) for x in this_expiries_dts.tolist()],
        "strikes": this_strikes.tolist()
    }

    json_path = Path.cwd() / "nn_learning_intensity" / "trainset" / f"json_data_{idx}"

    with open(json_path, "w") as f:
        json.dump(json_data, f)

"""
python nn_learning_intensity/make_trainset.py
"""

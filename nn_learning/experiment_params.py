from pathlib import Path
import numpy as np
from cross_excitation import cross_excitation
from param_class import CrossExcitation
from debug import Debugger
import json

name = "set_3"

save = Path.cwd() / "nn_learning" / "training_data" / name
save.mkdir(exist_ok=True)
dmode = 1
debugger = Debugger(mode=dmode)

json_path = Path.cwd() / "nn_learning" / "training_data" / "param_info.json"
if json_path.exists() and json_path.stat().st_size > 0:
    with open(json_path, "r") as f:
        json_data = json.load(f)
else:
    json_data = {}

params = CrossExcitation(
    # TIME SCALE
    dt=100_000,
    T=23,

    # STATIC BASE INTENSITY (per contract)
    alpha_moneyness = 0.15,
    alpha_time      =  2e-5,
    mu_intensity    = 4e-5,
    mu_variation = 0.5,

    # HAWKES DYNAMICS
    beta     = 0.00005,           # slower decay over ~1–2 days
    rho_self = 0.5,          # self-excitation < 1
    tau      = [[0.05, 0.02],  # weaker cross-excitation
                [0.02, 0.05]],
    gamma_m  = 2.0,            # smoother in strikes
    gamma_t  = 1e-3,            # smoother in expiries

    w_volume = 0.05,           # lower volume impact

    # VOLUME MODEL (unchanged for now, tune after eyeballing data)
    contract_volume_mean = 3.0,
    contract_volume_std  = 1.2,
    volume_base          = 3.0,
    volume_moneyness     = 0.5,
    volume_time_decay    = 0.002,

    # OPTION GRID
    strike_prices = [3800, 3900, 4000, 4100, 4200, 4300, 4400],
    expiry_dts    = [86400 * k for k in [5, 15, 30, 45]],

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

    init_open_price = 120.0,
    init_vola       = 0.04,

    # HESTON
    kappa = 1.8,
    theta = 0.04,
    xi    = 0.25,
    mu    = 0.06,
    rho   = -0.7,

    # ORDER TYPE LOGIC (unchanged)
    limit_order_base_param    = 1.2,
    limit_order_vol_param     = -0.3,
    limit_order_distance_param= 0.5,
    limit_order_spread_param  = 0.5,

    buy_order_base_param      = 0.0,
    buy_order_imbalance_param = 1.5
)

json_data[name] = params.model_dump_json()

cross_excitation(params, save=True, savedir=save, debugger=debugger)

with open(json_path, "w") as f:
    json.dump(json_data, f)

"""
python experiment_params.py
"""

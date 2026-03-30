import json
from pathlib import Path
from cross_excitation import cross_excitation
from param_class import CrossExcitation
from debug import Debugger
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import pandas as pd

# load real data
df = pd.read_csv(Path.cwd() / "data" / "spy_eod_202303.txt", index_col=0)
df.columns = [k.strip(' ') for k in df.columns]

T = 23

selected_strikes = [400.0, 410.0, 390.0, 380.0, 395.0, 405.0, 370.0]
selected_expiries = [' 2023-06-16', ' 2023-03-31', ' 2023-04-21', ' 2023-05-19']
selected_expiries = df[df["[EXPIRE_DATE]"].isin(selected_expiries)]["[EXPIRE_UNIX]"].unique()

def strip_values(val):
    new = val.strip(' ')
    if new == "":
        return 0.0
    return float(new)

real_volumes = np.zeros(T)
quotetimes = df.index.unique()
for t, quotetime in enumerate(quotetimes):
    section = df.loc[quotetime, :]
    section = section[section["[EXPIRE_UNIX]"].isin(selected_expiries)]
    section = section[section["[STRIKE]"].isin(selected_strikes)]
    section["[P_VOLUME]"] = section["[P_VOLUME]"].apply(strip_values)
    section["[C_VOLUME]"] = section["[C_VOLUME]"].apply(strip_values)
    volume = (section["[P_VOLUME]"] + section["[C_VOLUME]"]).sum()
    real_volumes[t] = volume

dmode = 1
debugger = Debugger(mode=dmode)

with open(Path.cwd() / "nn_learning" / "predicted_params.json", "r") as f:
    data = json.load(f)

params = CrossExcitation(
    # TIME SCALE
    dt=100_000,
    T=23,

    # STATIC BASE INTENSITY (per contract)
    alpha_moneyness = data["alpha_moneyness"],
    alpha_time      =  data["alpha_time"],
    mu_intensity    = data["mu"],
    mu_variation = 0.5,

    # HAWKES DYNAMICS
    beta     = data["beta"],           # slower decay over ~1–2 days
    rho_self = data["rho_self"],          # self-excitation < 1
    tau      = np.reshape(np.array(data["tau"]), (2, 2)),
    gamma_m  = data["gamma_m"],            # smoother in strikes
    gamma_t  = data["gamma_t"],            # smoother in expiries

    w_volume = data["w_volume"],           # lower volume impact

    # VOLUME MODEL (unchanged for now, tune after eyeballing data)
    contract_volume_mean = 3.0,
    contract_volume_std  = 1.2,
    volume_base          = data["volume_base"],
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

res = cross_excitation(params=params, save=False, debugger=debugger)
if res == False:
    raise

(
    orderbooks,
    assetdata,
    overviews,
    overviews_struct,
    trades,
    all_trades,
    intensities_keep,
    lambda_keep,
    num_events_keep,
    limit_probs,
    buys_probs,
    num_events_all_contracts,
    kernels,
    traded_volumes
) = res

simulated_volume = np.sum(traded_volumes, axis=(0, 1, 2)) # (T)

volume_fig = make_subplots(rows=2, cols=1)
volume_fig.add_trace(go.Scatter(x=list(range(T)), y=simulated_volume, name="Simulated volume (total)"), row=1, col=1)
volume_fig.add_trace(go.Scatter(x=list(range(T)), y=real_volumes, name="Real volume (total)"), row=2, col=1)
volume_fig.show()


"""
python nn_learning/evaluate_result.py
"""
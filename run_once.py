from param_class import Params
from engine import run
from pathlib import Path

params = Params(
dt=30.0,
T=390,
beta=2.0,
gamma_m=1.5,
gamma_t=0.5,
rho_self=0.8,
mu_intensity=0.01,
w=0.3,
tau=[[0.9, 0.3], [0.3, 0.9]],
contract_volume_mean=0.5,
contract_volume_std=0.8,
volume_base=1.0,
volume_moneyness=2.0,
volume_time_decay=1.5,
strike_prices=[4000, 4100, 4200, 4300, 4400, 4500],
expiry_dts=[86400*7, 86400*30, 86400*90],
risk_free=0.04,
dividend_rate=0.015,
base_n_orders_init=5,
base_scale_init_orders=0.005,
moneyness_scale_init_orders=3.0,
time_scale_init_orders=1.0,
beta_init=0.1,
gamma_init=0.5,
init_open_price=4200.0,
init_vola=0.04,
kappa=2.0,
theta=0.04,
xi=0.3,
mu=0.08,
rho=-0.7,
)


run(params, save=True, savedir=Path.cwd())
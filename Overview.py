import streamlit as st
from param_class import SelfExcitation
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

params = SelfExcitation(
dt=30.0,
T=100,
beta=0.005,
alpha_moneyness=100.0,
alpha_time=5.0,
mu_intensity=0.01,
w_volume=0.5,
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

st.set_page_config(
    page_title="Options Market Simulation Dashboard",
    layout="wide"
)

save = Path.cwd() / "run_self_excitation_result"


orderbooks = np.load(save / "orderbooks.npy", allow_pickle=True)
overview = np.load(save / "overviews.npy", allow_pickle=True)
trades = np.load(save / "trades.npy", allow_pickle=True)
all_trades = np.load(save / "all_trades.npy", allow_pickle=True)
assetdata = np.load(save / "assetdata.npy", allow_pickle=True)
intensities = np.load(save / "intensities_keep.npy", allow_pickle=True)
lambdas = np.load(save / "lambda_keep.npy", allow_pickle=True)
num_events = np.load(save / "num_events.npy", allow_pickle=True)

params_dict = params.model_dump()
param_df = pd.DataFrame({"Parameter": params_dict.keys(), "Values": params_dict.values()})
st.dataframe(param_df)

# Store data in session state
st.session_state.assetdata = assetdata
st.session_state.lambdas = lambdas
st.session_state.time_values = np.linspace(0, params.dt * params.T, int(params.dt))
st.session_state.all_trades = all_trades
st.session_state.intensities = intensities
st.session_state.expiry_dates = params.expiry_dts
st.session_state.strike_prices = params.strike_prices
st.session_state.num_events = num_events

price_fig = go.Figure()
vola_fig = go.Figure()

price_fig.add_trace(go.Scatter(x=st.session_state.time_values, y=st.session_state.assetdata[0, :]))
vola_fig.add_trace(go.Scatter(x=st.session_state.time_values, y=st.session_state.assetdata[1, :]))

st.write("Asset price graph")
st.plotly_chart(price_fig)

st.write("Asset volatility graph")
st.plotly_chart(vola_fig)
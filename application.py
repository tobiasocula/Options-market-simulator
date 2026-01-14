import streamlit as st
from engine import run
from param_class import Params
import pandas as pd
import plotly.graph_objects as go
import numpy as np

@st.cache_data
def run_simulation(params):
    # Call your simulation function
    return run(params)

# Initialize your Params object
# ----- Parameters -----
params = Params(
dt=30.0,
T=50,
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

st.set_page_config(
    page_title="Options Market Simulation Dashboard",
    layout="wide"
)

# Run the simulation
#orderbooks, assetdata, overviews, trades, all_trades, intensities = run_simulation(params)

orderbooks = np.load("orderbooks.npy", allow_pickle=True)
overview = np.load("overviews.npy", allow_pickle=True)
trades = np.load("trades.npy", allow_pickle=True)
all_trades = np.load("all_trades.npy", allow_pickle=True)
assetdata = np.load("assetdata.npy", allow_pickle=True)
intensities = np.load("intensities_keep.npy", allow_pickle=True)

# Store data in session state
st.session_state.assetdata = assetdata
st.session_state.timevalues = np.linspace(0, params.dt * params.T, int(params.T))
st.session_state.all_trades = all_trades
st.session_state.intensities = intensities
st.session_state.expiry_dates = params.expiry_dts
st.session_state.strike_prices = params.strike_prices

# Use a unique key for the slider
if 'slider_time' not in st.session_state:
    st.session_state.slider_time = 0

st.session_state.slider_time = st.slider(
    "Time Step",
    min_value=0,
    max_value=assetdata.shape[1]-1,
    value=st.session_state.slider_time,
    step=1,
    key="global_slider"
)

st.sidebar.success("Select a page above")
st.subheader(f"Time Step: {st.session_state.slider_time}")
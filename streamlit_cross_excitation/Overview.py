import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from params_for_cross_excitation import params

st.set_page_config(
    page_title="Options Market Simulation Dashboard",
    layout="wide"
)

save = Path.cwd() / "run_cross_excitation_result"

params_dict = params.model_dump()
param_df = pd.DataFrame({
    "Parameter": params_dict.keys(),
    "Values": [str(v) for v in params_dict.values()]
})

st.write("Parameters used for simulation")
st.dataframe(param_df)

# Store data in session state
st.session_state.assetdata = np.load(save / "assetdata.npy", allow_pickle=True)
st.session_state.lambdas = np.load(save / "lambda_keep.npy", allow_pickle=True)
st.session_state.overviews = np.load(save / "overviews.npy", allow_pickle=True)
st.session_state.time_values = np.linspace(0, params.dt * params.T, int(params.T))
st.session_state.all_trades = np.load(save / "all_trades.npy", allow_pickle=True)
st.session_state.intensities = np.load(save / "intensities_keep.npy", allow_pickle=True)
st.session_state.expiry_dates = params.expiry_dts
st.session_state.strike_prices = params.strike_prices
st.session_state.num_events = np.load(save / "num_events.npy", allow_pickle=True)
st.session_state.lim_probs = np.load(save / "limit_probs.npy", allow_pickle=True)
st.session_state.buy_probs = np.load(save / "buys_probs.npy", allow_pickle=True)
st.session_state.volumes = np.load(save / "volumes.npy", allow_pickle=True)
st.session_state.num_events_contracts = np.load(save / "num_events_contracts.npy", allow_pickle=True)
st.session_state.kernels = np.load(save / "kernels.npy", allow_pickle=True)

price_fig = go.Figure()
vola_fig = go.Figure()

price_fig.add_trace(go.Scatter(x=st.session_state.time_values, y=st.session_state.assetdata[0, :]))
vola_fig.add_trace(go.Scatter(x=st.session_state.time_values, y=st.session_state.assetdata[1, :]))

st.write("Asset price graph")
st.plotly_chart(price_fig)

st.write("Asset volatility graph")
st.plotly_chart(vola_fig)
import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import itertools
import pandas as pd

# change with different params
from cross_runs_scripts.params_long_1 import params

M = len(params.expiry_dts)
N = len(params.strike_prices)

st.subheader("Order activity & Trades")

buys_probs_time = np.nanmean(st.session_state.buy_probs, axis=(0, 1, 2))
limit_probs_time = np.nanmean(st.session_state.lim_probs, axis=(0, 1, 2))

buys_probs_time_fig = go.Figure()
limit_probs_time_fig = go.Figure()

buys_probs_time_fig.add_trace(go.Scatter(
    x=st.session_state.time_values,
    y=buys_probs_time,
    name=f"Buy probabilities over time (mean contracts)"
))
limit_probs_time_fig.add_trace(go.Scatter(
    x=st.session_state.time_values,
    y=limit_probs_time,
    name=f"Limit order probabilities over time (mean contracts)"
))

st.write(f"Buy probabilities over time (mean contracts)")
st.plotly_chart(buys_probs_time_fig)

st.write(f"Limit probabilities over time (mean contracts)")
st.plotly_chart(limit_probs_time_fig)
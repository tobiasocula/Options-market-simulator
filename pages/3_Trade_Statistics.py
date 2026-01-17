import streamlit as st
import plotly.express as px
from params_for_self_excitation import params
import numpy as np
import plotly.graph_objects as go
import itertools
import pandas as pd

M = len(params.expiry_dts)
N = len(params.strike_prices)

st.subheader("Order activity & Trades")

buys_probs_contracts = np.nanmean(st.session_state.buy_probs, axis=-1)
buys_probs_time = np.nanmean(st.session_state.buy_probs, axis=(0, 1, 2))

buys_probs_contracts_df_calls = pd.DataFrame(
    buys_probs_contracts[:, :, 0],
    columns=[str(x) for x in params.strike_prices],
    index=[str(x) for x in params.expiry_dts],
)

buys_probs_contracts_df_puts = pd.DataFrame(
    buys_probs_contracts[:, :, 1],
    columns=[str(x) for x in params.strike_prices],
    index=[str(x) for x in params.expiry_dts],
)

fig_calls_buys = px.imshow(
    buys_probs_contracts_df_calls,
    color_continuous_scale="viridis",
    zmin=buys_probs_contracts_df_calls.values.min(),
    zmax=buys_probs_contracts_df_calls.values.max(),
    labels=dict(x="Strike prices", y="Expiry dates"),
    aspect="auto"
)

fig_puts_buys = px.imshow(
    buys_probs_contracts_df_puts,
    color_continuous_scale="viridis",
    zmin=buys_probs_contracts_df_puts.values.min(),
    zmax=buys_probs_contracts_df_puts.values.max(),
    labels=dict(x="Strike prices", y="Expiry dates"),
    aspect="auto"
)

fig_puts_buys.update_yaxes(type="category")
fig_calls_buys.update_yaxes(type="category")

all_buy_probs = go.Figure()
for m, n, k in itertools.product(range(M), range(N), range(2)):
    all_buy_probs.add_trace(
        go.Scatter(
            x=st.session_state.time_values,
            y=st.session_state.buy_probs[m, n, k, :],
            name=f"Expiry {params.expiry_dts[m]}, Strike {params.strike_prices[n]}, Call: {k==0}"
        )
    )

buys_probs_time_fig = go.Figure()
buys_probs_time_fig.add_trace(go.Scatter(
    x=st.session_state.time_values,
    y=buys_probs_time,
    name=f"Buy probabilities over time (mean contracts)"
))

st.write("Distribution of buy/sell % of calls between contracts")
st.plotly_chart(fig_calls_buys, key="fig calls buys")
st.write("Distribution of buy/sell % of puts between contracts")
st.plotly_chart(fig_puts_buys, key="fig puts buys")
st.write("Buy/sell probabilities over time")
st.plotly_chart(all_buy_probs)
st.write(f"Buy probabilities over time (mean contracts)")
st.plotly_chart(buys_probs_time_fig)
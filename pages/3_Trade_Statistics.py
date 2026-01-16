import streamlit as st
import plotly.express as px
from params_for_self_excitation import params
import numpy as np
import plotly.graph_objects as go
import itertools

M = len(params.expiry_dts)
N = len(params.strike_prices)

st.subheader("Order activity & Trades")

buys_probs_contracts = np.nanmean(st.session_state.buy_probs, axis=-1)

fig_calls_buys = px.imshow(
    buys_probs_contracts[:, :, 0],
    color_continuous_scale="viridis",
    zmin=np.nanmin(buys_probs_contracts[:, :, 0]),
    zmax=np.nanmax(buys_probs_contracts[:, :, 0])
)

fig_puts_buys = px.imshow(
    buys_probs_contracts[:, :, 1],
    color_continuous_scale="viridis",
    zmin=np.nanmin(buys_probs_contracts[:, :, 1]),
    zmax=np.nanmax(buys_probs_contracts[:, :, 1])
)

all_buy_probs = go.Figure()
for m, n, k in itertools.product(range(M), range(N), range(2)):
    all_buy_probs.add_trace(
        go.Scatter(
            x=st.session_state.time_values,
            y=st.session_state.buy_probs[m, n, k, :],
            name=f"Expiry {params.expiry_dts[m]}, Strike {params.strike_prices[n]}, Call: {k==0}"
        )
    )

st.write("Distribution of buy/sell % of calls between contracts")
st.plotly_chart(fig_calls_buys, key="fig calls buys")
st.write("Distribution of buy/sell % of puts between contracts")
st.plotly_chart(fig_puts_buys, key="fig puts buys")
st.write("Buy/sell probabilities over time")
st.plotly_chart(all_buy_probs)
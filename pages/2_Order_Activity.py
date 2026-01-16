import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import numpy as np
import itertools
from params_for_self_excitation import params

st.subheader("Order activity & Trades")

intensities_mean_bids = pd.DataFrame(
    np.mean(st.session_state.intensities[:, :, 0, :], axis=-1),
    columns=params.strike_prices,
    index=params.expiry_dts
)

intensities_mean_asks = pd.DataFrame(
    np.mean(st.session_state.intensities[:, :, 1, :], axis=-1),
    columns=params.strike_prices,
    index=params.expiry_dts
)

M = len(params.expiry_dts)
N = len(params.strike_prices)

all_intensities = go.Figure()
for m, n, k in itertools.product(range(M), range(N), range(2)):
    all_intensities.add_trace(
        go.Scatter(
            x=st.session_state.time_values,
            y=st.session_state.intensities[m, n, k, :],
            name=f"Expiry {params.expiry_dts[m]}, Strike {params.strike_prices[n]}, Call: {k==0}"
        )
    )

fig_bids_mean = px.imshow(
    intensities_mean_bids,
    color_continuous_scale="viridis",
    zmin=intensities_mean_bids.values.min(),
    zmax=intensities_mean_bids.values.max()
)
print('intensities_mean_bids:', intensities_mean_bids)

fig_asks_mean = px.imshow(
    intensities_mean_asks,
    color_continuous_scale="viridis",
    zmin=intensities_mean_asks.values.min(),
    zmax=intensities_mean_asks.values.max()
)

lambda_fig = go.Figure()
lambda_fig.add_trace(go.Scatter(
    x=st.session_state.time_values,
    y=st.session_state.lambdas))

num_events_bar = px.bar(st.session_state.num_events)

st.write("All trades")

alltrades_df = pd.DataFrame(st.session_state.all_trades[-1],
    columns=["price", "time", "volume", "expiry", "strike", "call/put"])

st.dataframe(alltrades_df)

st.write("Trading intensities (mean of bids)")
st.plotly_chart(fig_bids_mean, key="bids_mean")

st.write("Trading intensities (mean of asks)")
st.plotly_chart(fig_asks_mean, key="asks_mean")

st.write("Intensities over time (combined)")
st.plotly_chart(lambda_fig)

st.write("Intensities of all contracts over time")
st.plotly_chart(all_intensities)

st.write("Number of orders over time (total)")
st.plotly_chart(num_events_bar)
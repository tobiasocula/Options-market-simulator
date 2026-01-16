import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import numpy as np

st.subheader("Order activity & Trades")

intensities_mean_bids = np.mean(st.session_state.intensities[:, :, 0, :], axis=-1)
intensities_mean_asks = np.mean(st.session_state.intensities[:, :, 1, :], axis=-1)


fig_bids_mean = px.imshow(
    intensities_mean_bids,
    color_continuous_scale="viridis",
    zmin=intensities_mean_bids.min(),
    zmax=intensities_mean_bids.max()
)

fig_asks_mean = px.imshow(
    intensities_mean_asks,
    color_continuous_scale="viridis",
    zmin=intensities_mean_asks.min(),
    zmax=intensities_mean_asks.max()
)

lambda_fig = go.Figure()
lambda_fig.add_trace(go.Scatter(
    x=st.session_state.time_values,
    y=st.session_state.lambdas))

num_events_bar = px.bar(st.session_state.num_events)

st.write("Trading intensities (mean of bids)")
st.plotly_chart(fig_bids_mean, key="bids_mean")

st.write("Trading intensities (mean of asks)")
st.plotly_chart(fig_asks_mean, key="asks_mean")

st.write("Intensities over time (combined)")
st.plotly_chart(lambda_fig)

st.write("Number of orders over time (total)")
st.plotly_chart(num_events_bar)
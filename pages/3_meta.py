import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import plotly.express as px
import numpy as np

if "assetdata" not in st.session_state or "timevalues" not in st.session_state:
    st.error("Run simulation from main page first.")
    st.stop()

st.subheader("Meta")

# Use the same slider, bound to session state
st.session_state.slider_time = st.slider(
    "Time Step",
    min_value=0,
    max_value=st.session_state.assetdata.shape[1]-1,
    value=st.session_state.slider_time,
    step=1,
    key="global_slider"
)

bid_table = pd.DataFrame(
    st.session_state.intensities[:, :, 0, st.session_state.slider_time],
    index=st.session_state.expiry_dates,
    columns=st.session_state.strike_prices
    )

ask_table = pd.DataFrame(
    st.session_state.intensities[:, :, 1, st.session_state.slider_time],
    index=st.session_state.expiry_dates,
    columns=st.session_state.strike_prices
    )

intensities_mean_bids = np.mean(st.session_state.intensities[:, :, 0, :], axis=-1)
intensities_mean_asks = np.mean(st.session_state.intensities[:, :, 1, :], axis=-1)

intensities_mean_asks_df = pd.DataFrame(
    intensities_mean_asks,
    index=st.session_state.expiry_dates,
    columns=st.session_state.strike_prices
)

intensities_mean_bids_df = pd.DataFrame(
    intensities_mean_bids,
    index=st.session_state.expiry_dates,
    columns=st.session_state.strike_prices
)

fig_bids = px.imshow(ask_table)
fig_asks = px.imshow(bid_table)

print('ask table intensities:'); print(ask_table)
print('bid table intensities:'); print(bid_table)

fig_bids_mean = px.imshow(intensities_mean_bids_df)
fig_asks_mean = px.imshow(intensities_mean_asks_df)

print('bid mean intensities:', intensities_mean_bids_df)
print('asks mean intensities:', intensities_mean_asks_df)

st.write("Trading intensities per timestamp (bids)")
st.plotly_chart(fig_bids, key="bids_current")

st.write("Trading intensities per timestamp (asks)")
st.plotly_chart(fig_asks, key="asks_current")

st.write("Trading intensities (mean of bids)")
st.plotly_chart(fig_bids_mean, key="bids_mean")

st.write("Trading intensities (mean of asks)")
st.plotly_chart(fig_asks_mean, key="asks_mean")
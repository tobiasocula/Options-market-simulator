import streamlit as st
import plotly.express as px
from params_for_self_excitation import params
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import itertools

st.subheader("Trade Volumes")

M = len(params.expiry_dts)
N = len(params.strike_prices)

volumes_contracts = np.nanmean(st.session_state.volumes, axis=-1) # (M, N, 2)

vols_calls_df = pd.DataFrame(
    volumes_contracts[:, :, 0],
    columns=[str(x) for x in params.strike_prices],
    index=[str(x) for x in params.expiry_dts],
)

vols_puts_df = pd.DataFrame(
    volumes_contracts[:, :, 1],
    columns=[str(x) for x in params.strike_prices],
    index=[str(x) for x in params.expiry_dts],
)

vols_calls = px.imshow(
    vols_puts_df,
    color_continuous_scale="viridis",
    zmin=vols_puts_df.values.min(),
    zmax=vols_puts_df.values.max(),
    labels=dict(x="Strike prices", y="Expiry dates"),
    aspect="auto"
)

vols_puts = px.imshow(
    vols_calls_df,
    color_continuous_scale="viridis",
    zmin=vols_calls_df.values.min(),
    zmax=vols_calls_df.values.max(),
    labels=dict(x="Strike prices", y="Expiry dates"),
    aspect="auto"
)

vols_calls.update_yaxes(type="category")
vols_puts.update_yaxes(type="category")

st.write("Volumes over time (call contracts)")
st.plotly_chart(vols_calls)
st.write("Volumes over time (put contracts)")
st.plotly_chart(vols_puts)

vols_time = go.Figure()
for m, n, k in itertools.product(range(M), range(N), range(2)):
    vols_time.add_trace(go.Scatter(
        x=st.session_state.time_values,
        y=st.session_state.volumes[m, n, k, :],
        name=f"Expiry {params.expiry_dts[m]}, Strike {params.strike_prices[n]}, Call: {k==0}"
    ))

st.write("Volumes over time")
st.plotly_chart(vols_time)
import streamlit as st
import plotly.express as px
from params_for_self_excitation import params
import numpy as np
import plotly.graph_objects as go
import itertools

st.subheader("Trade Volumes")

M = len(params.expiry_dts)
N = len(params.strike_prices)

volumes_contracts = np.nanmean(st.session_state.volumes, axis=-1) # (M, N, 2)

vols_calls = px.imshow(
    volumes_contracts[:, :, 0],
    color_continuous_scale="viridis",
    zmin=np.nanmin(volumes_contracts[:, :, 0]),
    zmax=np.nanmax(volumes_contracts[:, :, 0])
)

vols_puts = px.imshow(
    volumes_contracts[:, :, 1],
    color_continuous_scale="viridis",
    zmin=np.nanmin(volumes_contracts[:, :, 1]),
    zmax=np.nanmax(volumes_contracts[:, :, 1])
)

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

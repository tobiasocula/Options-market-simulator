
import streamlit as st
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import json
import itertools

# streamlit run nn_learning/streamlit/Main.py

st.set_page_config(
    page_title="Options Market Simulation Dashboard",
    layout="wide"
)

dir = Path.cwd() / "nn_learning" / "training_data"
dirs = [k for k in dir.iterdir() if k.name != "param_info.json"]
datadir = Path.cwd() / "nn_learning" / "training_data"

with open(dir / "param_info.json", "r") as f:
    json_data = json.load(f)

col1, col2 = st.columns(2)
for run in json_data.keys():

    paramset = json.loads(json_data[run])
    datadir = dir / run

    M = len(paramset["expiry_dts"])
    N = len(paramset["strike_prices"])

    chosen_vol = np.load(datadir / "traded_volumes.npy", allow_pickle=True) # (M, N, 2, T)
    mean_volumes = np.mean(chosen_vol, axis=(0, 1, 2)) # T

    with col1:
        for key in ["gamma_m", "gamma_t", "mu_intensity", "beta", "w_volume"]:
            st.write(f"{key}: {paramset[key]}")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(23)), y=mean_volumes, name="Volumes"))
        st.plotly_chart(fig)

    with col2:

        fig = go.Figure()
        for m, n, k in itertools.product(range(M), range(N), range(2)):
            expiry = paramset["expiry_dts"][m]
            strike = paramset["strike_prices"][n]
            name = "Contract {}, {}, call: {}".format(expiry, strike, k)
            fig.add_trace(go.Scatter(x=list(range(23)), y=chosen_vol[m, n, k, :], name=name))

        st.plotly_chart(fig)

"""
streamlit run nn_learning/streamlit/Main.py
"""
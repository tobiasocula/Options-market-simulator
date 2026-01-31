import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import numpy as np
import itertools

# change with different params
from cross_runs_scripts.params_long_1 import params

st.subheader("Overview")

# Slider to set a parameter
parameter_value = st.slider(
    "Select time for order book overview:",
    min_value=0.0,
    max_value=float(params.T),
    value=0.0,
    step=1.0,
    help="Adjust this slider to change the parameter value."
)

selected = st.session_state.overviews[int(parameter_value)]
df = pd.DataFrame(selected)

st.write("Overview of orderbook")

st.dataframe(df)
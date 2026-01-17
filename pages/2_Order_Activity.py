import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import numpy as np
import itertools
from params_for_self_excitation import params

st.subheader("Order activity & Trades")

M = len(params.expiry_dts)
N = len(params.strike_prices)

calls_intensities = np.nanmean(st.session_state.intensities[:, :, 0, :], axis=-1)
puts_intensities = np.nanmean(st.session_state.intensities[:, :, 1, :], axis=-1)

calls_intensities_df = pd.DataFrame(
    calls_intensities,
    columns=[str(x) for x in params.strike_prices],
    index=[str(x) for x in params.expiry_dts],
)

puts_intensities_df = pd.DataFrame(
    puts_intensities,
    columns=[str(x) for x in params.strike_prices],
    index=[str(x) for x in params.expiry_dts],
)

all_intensities = go.Figure()
for m, n, k in itertools.product(range(M), range(N), range(2)):
    all_intensities.add_trace(
        go.Scatter(
            x=st.session_state.time_values,
            y=st.session_state.intensities[m, n, k, :],
            name=f"Expiry {params.expiry_dts[m]}, Strike {params.strike_prices[n]}, Call: {k==0}"
        )
    )

fig_calls_intensities = px.imshow(
    calls_intensities_df,
    color_continuous_scale="viridis",
    zmin=calls_intensities_df.values.min(),
    zmax=calls_intensities_df.values.max(),
    labels=dict(x="Strike prices", y="Expiry dates"),
    aspect="auto"
)

fig_puts_intensities = px.imshow(
    puts_intensities_df,
    color_continuous_scale="viridis",
    zmin=puts_intensities_df.values.min(),
    zmax=puts_intensities_df.values.max(),
    labels=dict(x="Strike prices", y="Expiry dates"),
    aspect="auto"
)

fig_puts_intensities.update_yaxes(type="category")
fig_calls_intensities.update_yaxes(type="category")

print('fig puts intensities:'); print(fig_puts_intensities)
print('associated df:'); print(puts_intensities_df)

lambda_fig = go.Figure()
lambda_fig.add_trace(go.Scatter(
    x=st.session_state.time_values,
    y=st.session_state.lambdas))

num_events_bar = px.bar(st.session_state.num_events)

events_per_contract = go.Figure()
#for m, n, k in itertools.product(range(M), range(N), range(2)):

st.write("All trades")

alltrades_df = pd.DataFrame(st.session_state.all_trades[-1],
    columns=["price", "time", "volume", "expiry", "strike", "call/put"])

st.dataframe(alltrades_df)

st.write("Trading intensities (mean of calls)")
st.plotly_chart(fig_calls_intensities, key="calls_mean")

st.write("Trading intensities (mean of puts)")
st.plotly_chart(fig_puts_intensities, key="puts_mean")

st.write("Intensities over time (combined)")
st.plotly_chart(lambda_fig)

st.write("Intensities of all contracts over time")
st.plotly_chart(all_intensities)

st.write("Number of orders over time (total)")
st.plotly_chart(num_events_bar)
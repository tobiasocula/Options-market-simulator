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

intensities_wanted = np.nanmean(st.session_state.intensities, axis=-1)
kernels_wanted = np.nanmean(st.session_state.kernels, axis=-1)

calls_intensities_df = pd.DataFrame(
    intensities_wanted[:, :, 0],
    columns=[str(x) for x in params.strike_prices],
    index=[str(x) for x in params.expiry_dts],
)

puts_intensities_df = pd.DataFrame(
    intensities_wanted[:, :, 1],
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

kernels_calls_df = pd.DataFrame(
    kernels_wanted[:, :, 0],
    columns=[str(x) for x in params.strike_prices],
    index=[str(x) for x in params.expiry_dts]
)

kernels_puts_df = pd.DataFrame(
    kernels_wanted[:, :, 1],
    columns=[str(x) for x in params.strike_prices],
    index=[str(x) for x in params.expiry_dts]
)

kernels_calls = px.imshow(
    kernels_calls_df,
    color_continuous_scale="viridis",
    zmin=kernels_calls_df.values.min(),
    zmax=kernels_calls_df.values.max(),
    labels=dict(x="Strike prices", y="Expiry dates"),
    aspect="auto"
)

kernels_puts = px.imshow(
    kernels_puts_df,
    color_continuous_scale="viridis",
    zmin=kernels_puts_df.values.min(),
    zmax=kernels_puts_df.values.max(),
    labels=dict(x="Strike prices", y="Expiry dates"),
    aspect="auto"
)

fig_puts_intensities.update_yaxes(type="category")
fig_calls_intensities.update_yaxes(type="category")

kernels_puts.update_yaxes(type="category")
kernels_calls.update_yaxes(type="category")

lambda_fig = go.Figure()
lambda_fig.add_trace(go.Scatter(
    x=st.session_state.time_values,
    y=st.session_state.lambdas,
    name="Excitations"))

lambda_fig.add_trace(go.Bar(
    x=st.session_state.time_values,
    y=st.session_state.num_events,
    name="Number of orders",
    yaxis="y2"))

lambda_fig.update_layout(
    xaxis_title="Time",
    yaxis=dict(title="Excitations"),  # Primary y-axis label
    yaxis2=dict(
        title="Number of orders",
        overlaying="y",
        side="right"
    )
)

st.write("All trades")

max_trade_count = min(100, len(st.session_state.all_trades))

alltrades_df = pd.DataFrame(st.session_state.all_trades[:max_trade_count],
    columns=["price", "time", "volume", "expiry", "strike", "call/put"])

st.dataframe(alltrades_df)

print(kernels_calls_df)

st.write("Trading intensities (mean of calls)")
st.plotly_chart(fig_calls_intensities, key="calls_mean")

st.write("Trading intensities (mean of puts)")
st.plotly_chart(fig_puts_intensities, key="puts_mean")

st.write("Intensities over time (combined)")
st.plotly_chart(lambda_fig, key="intensities_comb")

st.write("Intensities of all contracts over time")
st.plotly_chart(all_intensities, key="intensities_contr")

st.write("Kernels of cross-excitation between contracts")
st.write("Call contracts")
st.plotly_chart(kernels_calls, key="kernels_calls")
st.write("Put contracts")
st.plotly_chart(kernels_puts, key="kernels_puts")
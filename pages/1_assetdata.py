import streamlit as st
import plotly.graph_objects as go

if "assetdata" not in st.session_state or "timevalues" not in st.session_state:
    st.error("Run simulation from main page first.")
    st.stop()

st.subheader("Asset Data")

# Use the same slider, bound to session state
st.session_state.slider_time = st.slider(
    "Time Step",
    min_value=0,
    max_value=st.session_state.assetdata.shape[1]-1,
    value=st.session_state.slider_time,
    step=1,
    key="global_slider"
)

price_fig = go.Figure()
vola_fig = go.Figure()

price_fig.add_trace(go.Scatter(x=st.session_state.timevalues, y=st.session_state.assetdata[0, :st.session_state.slider_time+1]))
vola_fig.add_trace(go.Scatter(x=st.session_state.timevalues, y=st.session_state.assetdata[1, :st.session_state.slider_time+1]))

st.write("Asset price graph")
st.plotly_chart(price_fig)

st.write("Asset volatility graph")
st.plotly_chart(vola_fig)

st.write(f"Asset Price: {st.session_state.assetdata[0, st.session_state.slider_time]:.2f}")
st.write(f"Asset Volatility: {st.session_state.assetdata[1, st.session_state.slider_time]:.2f}")

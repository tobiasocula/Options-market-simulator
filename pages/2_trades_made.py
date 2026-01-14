import streamlit as st
import plotly.graph_objects as go
import pandas as pd

if "assetdata" not in st.session_state or "timevalues" not in st.session_state:
    st.error("Run simulation from main page first.")
    st.stop()

st.subheader("Trades made")

# Use the same slider, bound to session state
st.session_state.slider_time = st.slider(
    "Time Step",
    min_value=0,
    max_value=st.session_state.assetdata.shape[1]-1,
    value=st.session_state.slider_time,
    step=1,
    key="global_slider"
)

st.write("All trades")

alltrades_df = pd.DataFrame(st.session_state.all_trades[st.session_state.slider_time],
    columns=["price", "time", "volume", "expiry", "strike", "call/put"])

print('all trades made:'); print(alltrades_df)

st.dataframe(alltrades_df)

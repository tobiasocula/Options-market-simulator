import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# if "assetdata" not in st.session_state or "time_values" not in st.session_state:
#     st.error("Run simulation from main page first.")
#     st.stop()

st.subheader("Trades made")

st.write("All trades")

alltrades_df = pd.DataFrame(st.session_state.all_trades[-1],
    columns=["price", "time", "volume", "expiry", "strike", "call/put"])

print('all trades made:'); print(alltrades_df)

st.dataframe(alltrades_df)

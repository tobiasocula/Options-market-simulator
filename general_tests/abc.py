import plotly.graph_objects as go
import numpy as np

def dist(x, param):
    return np.random.uniform(0, param) * x + 2.0

def inverse_cdf(u):

fig = go.Figure()
fig.add_trace(go.Histogram(data))
fig.show()


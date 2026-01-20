import numpy as np
import plotly.graph_objects as go
import pandas as pd

b = go.Figure()

x = np.linspace(0, 100, 10)
y = np.random.uniform(0, 1, 10)
b.add_trace(go.Bar(x=x, y=y))
b.update_layout(xaxis_title="x", yaxis_title="y")
b.show()

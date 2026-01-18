import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

import numpy as np

a = np.random.uniform(0, 1, 100)
df = pd.DataFrame(a)

#fig = px.bar(df, x=df.index, y=[str(k) for k in range(3)], barmode="group")
fig = go.Figure()
fig.add_trace(go.Bar(y=a, x=np.linspace(0, 10, a.shape[0])))
fig.add_trace(go.Scatter(x=[0, 0], y=[0, 1]))
fig.show()
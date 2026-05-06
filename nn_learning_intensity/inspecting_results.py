from plotly.subplots import make_subplots
from pathlib import Path
import plotly.graph_objects as go
import json


path = Path.cwd() / "nn_learning_intensity" / "results"

with open(path / "training_history.json", "r") as f:
    history = json.load(f)
print('keys:')
print(history.keys())


fig = make_subplots(rows=2, cols=2)

fig.add_trace(go.Scatter(x=list(range(len(history["mu_loss"]))), y=history["mu_loss"]), row=1, col=1)
fig.add_trace(go.Scatter(x=list(range(len(history["val_mu_loss"]))), y=history["val_mu_loss"], line=dict(dash="dash")), row=1, col=1)

fig.add_trace(go.Scatter(x=list(range(len(history["alpha_moneyness_loss"]))), y=history["alpha_moneyness_loss"]), row=1, col=2)
fig.add_trace(go.Scatter(x=list(range(len(history["val_alpha_moneyness_loss"]))), y=history["val_alpha_moneyness_loss"], line=dict(dash="dash")), row=1, col=2)

fig.add_trace(go.Scatter(x=list(range(len(history["alpha_time_loss"]))), y=history["alpha_time_loss"]), row=2, col=1)
fig.add_trace(go.Scatter(x=list(range(len(history["val_alpha_time_loss"]))), y=history["val_alpha_time_loss"], line=dict(dash="dash")), row=2, col=1)

fig.add_trace(go.Scatter(x=list(range(len(history["beta_loss"]))), y=history["beta_loss"]), row=2, col=2)
fig.add_trace(go.Scatter(x=list(range(len(history["val_beta_loss"]))), y=history["val_beta_loss"], line=dict(dash="dash")), row=2, col=2)

fig.show()
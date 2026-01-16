from pydantic import BaseModel
import plotly.express as px
import numpy as np
import pandas as pd

a = {"a": 2, "b": 1}

print(pd.DataFrame(a.values(), index=a.keys(), columns=["values"]).reset_index())

k = pd.DataFrame(
    {"values": a.values(),
     "keys": a.keys()}
)
print(k)
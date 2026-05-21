"""
needed datastructure for feeding model:
(T, num_contracts, 4)

use strikes around 130

"""

import pandas as pd
import numpy as np
from pathlib import Path

first_quote = pd.to_datetime("2023-03-01 16:00:00") # treat as starting date
last_quote = pd.to_datetime("2023-03-31 16:00:00") # treat as last date
ref_diff = (last_quote - first_quote).total_seconds()

expiries_used = np.load(Path.cwd() / "nn_learning_intensity" / "expiries_being_used.npy", allow_pickle=True)
strikes_used = np.load(Path.cwd() / "nn_learning_intensity" / "strikes_being_used.npy", allow_pickle=True)

expiries_used = [x[1:] for x in expiries_used]
expiries_used = pd.to_datetime(expiries_used)
print('here')
print(expiries_used)
print(len(expiries_used), len(strikes_used))
print(strikes_used)
print()


# load real data
df = pd.read_csv(Path.cwd() / "data" / "spy_eod_202303.txt", index_col=0)
print(df.columns)
g = df.groupby([" [QUOTE_READTIME]"])
groups = [x[1:] for x in g.groups]
groups = sorted(pd.to_datetime(groups))
print(groups)

groups = [(exp - first_quote).total_seconds() / ref_diff for exp in groups]

print(groups)
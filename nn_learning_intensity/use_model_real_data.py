"""
needed datastructure for feeding model:
(T, num_contracts, 4)

use strikes around 130

"""

import pandas as pd
import numpy as np
from pathlib import Path

expiries_used = np.load(Path.cwd() / "nn_learning_intensity" / "expiries_being_used.npy", allow_pickle=True)
strikes_used = np.load(Path.cwd() / "nn_learning_intensity" / "strikes_being_used.npy", allow_pickle=True)

expiries_used = [x[1:] for x in expiries_used]
expiries_used = pd.to_datetime(expiries_used)
print(expiries_used)
print()


# load real data
df = pd.read_csv(Path.cwd() / "data" / "spy_eod_202303.txt", index_col=0)
print(df.columns)
#df_filtered = df[(df[" [STRIKE]"].isin(strikes_used)) & df[" [EXPIRE_DATE]"].isin(expiries_used)]
df_filtered = df[df[" [STRIKE]"].isin(strikes_used)]
print(df_filtered)
import numpy as np
from pathlib import Path

dir = Path.cwd() / "nn_learning" / "training_data" / "set_0" / "assetdata.npy"

data = np.load(dir, allow_pickle=True)
print(data.shape)
"""
cut off time layers on all data within training_data (from T=25 to T=23)
"""

from pathlib import Path
import numpy as np
import sys

dir = Path.cwd() / "nn_learning" / "training_data"
for d in dir.iterdir():
    if d.name == "param_info.json":
        continue

    assetdata = np.load(dir / d.name / "assetdata.npy", allow_pickle=True)
    buys_probs = np.load(dir / d.name / "buys_probs.npy", allow_pickle=True)
    intensities_keep = np.load(dir / d.name / "intensities_keep.npy", allow_pickle=True)
    kernels = np.load(dir / d.name / "kernels.npy", allow_pickle=True)
    
    
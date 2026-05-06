# import numpy as np

# a = np.array(range(10))
# b = np.array(range(10))

# idx = np.random.permutation(len(a))

# a_shuff = a[idx]
# b_shuff = b[idx]
# print(a_shuff)
# print(b_shuff)

from pathlib import Path

root = Path.cwd() / 'nn_learning_intensity' / 'trainset'
for d in root.iterdir():
    if d.is_file() and d.name[0] == "j":
        new_name = d.with_name(d.name + ".json")
        print(f"Renaming {d.name} -> {new_name.name}")
        d.rename(new_name)
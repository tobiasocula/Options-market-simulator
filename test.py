import numpy as np
import plotly.graph_objects as go
import pandas as pd

M = 4
N = 3
P = 2
n, m, p = 0, 0, 0
for k in range(M * N * P):
    if k % N == 0 and k != 0:
        m += 1
        n = 0
    if k == M * N:
        p += 1
        n = 0
        m = 0

    print(n, m, p, k)
    n += 1
import numpy as np
from scipy.special import binom
from itertools import product

kvalues = list(range(3, 10))
nvalues = list(range(3, 10))
num = []

for k,n in product(kvalues, nvalues):
    left = sum([binom(n, l) * l * (1/k)**l * (1-1/k)**(n-l) for l in range(1, n+1)])
    right = n / k
#    print('equal:', left == right)
    num.append(left == right)
    print(left, ':', right)

total = len(kvalues) * len(nvalues)
print('total:', total)
print('equal:', len([val for val in num if val]))

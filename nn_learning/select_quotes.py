#import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import pandas as pd
import itertools
import sys

df = pd.read_csv(Path.cwd() / "data" / "spy_eod_202303.txt", index_col=0)
df.columns = [k.strip(' ') for k in df.columns]
print(df.columns)
all_strikes = df["[STRIKE]"].unique()
all_expiries = df["[EXPIRE_DATE]"].unique()

start_day = df["[QUOTE_READTIME]"].min() # 2023-03-01 16:00
start_day_day = 1
start_day_month = 3



def k_choose(array, k):
    result = []
    for i in range(k):
        sample = np.random.choice(array)
        while sample in result:
            sample = np.random.choice(array)
        result.append(sample)
    return result



# n_iters = 100
# already_had = []
# num_quotes = []

# for _ in range(n_iters):
#     chosen_strike = np.random.choice(all_strikes)
#     chosen_expiry = np.random.choice(all_expiries)
#     while (chosen_strike, chosen_expiry) in already_had:
#         chosen_expiry = np.random.choice(all_expiries)
#         chosen_strike = np.random.choice(all_strikes)
#     already_had.append((chosen_strike, chosen_expiry))

#     # check amount of quotes for this pair
#     select = df[(df["[EXPIRE_DATE]"] == chosen_expiry) & (df["[STRIKE]"] == chosen_strike)]
#     num_quotes.append(len(select))

# combined = [[count, pair] for count, pair in zip(num_quotes, already_had)]
# combined.sort(key=lambda x: x[0])
# for item in combined:
#     print(item)
#     print()

"""
n_iters = 100
already_had = [] # pairs
num_quotes = []

for _ in range(n_iters):
    chosen_expiries = k_choose(all_expiries, 4)
    chosen_strikes = k_choose(all_strikes, 4)
    already_had.append((chosen_expiries, chosen_strikes))
    while (chosen_strikes, chosen_expiries) in already_had:
        chosen_expiries = k_choose(all_expiries, 4)
        chosen_strikes = k_choose(all_strikes, 4)
    already_had.append((chosen_strikes, chosen_expiries))
    for expiry, strike in zip(chosen_expiries, chosen_strikes):
        select = df[(df["[EXPIRE_DATE]"] == expiry) & (df["[STRIKE]"] == strike)]
        num_quotes.append(len(select))

combined = [[count, pair] for count, pair in zip(num_quotes, already_had)]
combined.sort(key=lambda x: x[0])
for item in combined:
    print(item)
    print()

"""

selected_strikes = [120.0, 130.0, 140.0, 150.0]

n_iters = 100
already_had = []
num_quotes = []

for _ in range(n_iters):
    chosen_expiries = k_choose(all_expiries, 4)
    already_had.append(chosen_expiries)
    while chosen_expiries in already_had:
        chosen_expiries = k_choose(all_expiries, 4)
    already_had.append(chosen_expiries)
    # for expiry, strike in zip(chosen_expiries, selected_strikes):
    #     select = df[(df["[EXPIRE_DATE]"] == expiry) & (df["[STRIKE]"] == strike)]
    #     num_quotes.append(len(select))
    count = 0
    for expiry, strike in itertools.product(chosen_expiries, selected_strikes):
        select = df[(df["[EXPIRE_DATE]"] == expiry) & (df["[STRIKE]"] == strike)]
        if len(select) > 0:
            count += 1

    num_quotes.append(count)
    

combined = [[count, pair] for count, pair in zip(num_quotes, already_had)]

def day_differences(array):
    days_diff_arr = []
    for date in array:
        temp = date.split('-')
        days_diff = abs(int(temp[2]) - start_day_day)
        months_diff = abs(int(temp[1]) - start_day_month)
        days = days_diff + 30 * months_diff
        days_diff_arr.append(days)
    return days_diff_arr


combined.sort(key=lambda x: x[0])
combined_23pairs = combined

print(combined_23pairs)

"""
[[23, [' 2023-05-19', ' 2023-04-03', ' 2023-03-24', ' 2023-03-01']],
[23, [' 2023-03-28', ' 2023-04-06', ' 2023-03-21', ' 2023-03-13']],
[23, [' 2023-03-08', ' 2023-03-03', ' 2023-03-13', ' 2023-03-28']],
[23, [' 2023-03-22', ' 2023-03-10', ' 2023-03-31', ' 2023-03-17']],
[23, [' 2024-06-21', ' 2023-04-03', ' 2023-03-02', ' 2023-03-21']],
[23, [' 2023-10-20', ' 2023-04-03', ' 2025-03-21', ' 2023-05-12']],
[23, [' 2023-03-08', ' 2023-06-16', ' 2023-04-06', ' 2023-05-19']], [23, [' 2023-12-29', ' 2023-03-13', ' 2025-03-21', ' 2023-03-01']]]

"""

def error_term(pair):
    dates = pair[1]
    err = 0.0
    wanted = [5, 15, 30, 45]
    for w, date in zip(wanted, dates):
        items = date.split('-')
        days_diff = abs(int(items[2]) - start_day_day)
        months_diff = abs(int(items[1]) - start_day_month)
        diff = days_diff + 30*months_diff
        err += ((diff - w)/w)**2
    return err

combined_23pairs.sort(key=lambda x: (x[0], error_term(x)))

# [23, [' 2023-03-06', ' 2023-03-13', ' 2023-04-13', ' 2023-03-22']]
for item in combined_23pairs:
    print(item)
    print()
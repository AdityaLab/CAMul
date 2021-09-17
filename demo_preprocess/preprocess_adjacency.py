import numpy as np
import pandas as pd
import pickle
import os

data = np.loadtxt("./data/demographics/state_adjacency.csv", delimiter=",", dtype=str)[1:]
states = list(pd.read_csv("./data/demographics/State_codes.csv")["Code"])

adj_list = {st: set() for st in states}
for u, v in data:
    adj_list[u].add(v)
    adj_list[v].add(u)

os.makedirs("./data/demographics/saves", exist_ok=True)
with open("./data/demographics/saves/adj_list.pkl", "wb") as fl:
    pickle.dump(adj_list, fl)

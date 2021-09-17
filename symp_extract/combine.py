# Combine ILI and Symp data
import os
import pickle

import pandas as pd

from consts import include_cols

# Get ILI data
ili = pd.read_csv("./data/ili_data/ILINet.csv")

# Get symp data
with open("./data/symptom_data/saves/df.pkl", "rb") as f:
    symp = pickle.load(f)

ili = ili[["REGION", "YEAR", "WEEK", "% WEIGHTED ILI"]]


def region_to_hhs(x: str):
    if x == "X":
        return 0
    else:
        return int(x[-2:])


ili["hhs"] = ili["REGION"].apply(region_to_hhs)
ili["epiweek"] = ili["WEEK"]
ili["year"] = ili["YEAR"]
ili["ili"] = ili["% WEIGHTED ILI"]
symp = symp[["epiweek", "year", "hhs"] + include_cols]
symp = symp.groupby(["epiweek", "year", "hhs"])[include_cols].mean().reset_index()
ili = ili[["epiweek", "hhs", "year", "ili"]]
merge_df = symp.merge(ili, on=["epiweek", "year", "hhs"], how="left")

merge_df = merge_df.dropna()
with open("./data/symptom_data/saves/combine.pkl", "wb") as f:
    pickle.dump(merge_df, f)

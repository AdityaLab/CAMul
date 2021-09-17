import os
import pickle
from datetime import datetime
from glob import glob

import pandas as pd
from epiweeks import Week

from consts import hhs_regions, include_cols

files = sorted(glob("./data/symptom_data/US_*.csv"))
dfs = [pd.read_csv(f) for f in files]
dfs = [
    df[["country_region_code", "sub_region_1_code", "date"] + include_cols]
    for df in dfs
]
all_df = pd.concat(dfs)


def get_code(x: str):
    try:
        return x[-2:]
    except:
        return "US"


def date_to_epiweek(x: str, format="%Y-%m-%d"):
    d = datetime.strptime(x, format)
    w = Week.fromdate(d.date())
    return w.week


def date_to_year(x: str, format="%Y-%m-%d"):
    d = datetime.strptime(x, format)
    w = Week.fromdate(d.date())
    return w.year


all_df["geo_code"] = all_df["sub_region_1_code"].apply(get_code)
all_df["epiweek"] = all_df["date"].apply(date_to_epiweek)
all_df["year"] = all_df["date"].apply(date_to_year)


def state_to_hhs(state):
    if state == "US":
        return 0
    else:
        for i in range(1, 11):
            if state in hhs_regions[i]:
                return i


all_df["hhs"] = all_df["geo_code"].apply(state_to_hhs)

all_df = all_df[["year", "epiweek", "geo_code", "hhs"] + include_cols]
all_df = all_df.fillna(0)

os.makedirs("./data/symptom_data/saves", exist_ok=True)
with open("./data/symptom_data/saves/df.pkl", "wb") as f:
    pickle.dump(all_df, f)

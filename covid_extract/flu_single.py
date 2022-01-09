import numpy as np
import pandas as pd

from optparse import OptionParser
import pickle, os
from .hosp_consts import *


parser = OptionParser()
parser.add_option("-r", "--region", dest="region", default="X",)
parser.add_option("-e", "--epiweek", dest="epiweek", default="202113",)
parser.add_option("-s", "--smooth", dest="smooth", default=1, type="int")
(options, args) = parser.parse_args()

region = options.region
epiweek = options.epiweek
smooth = options.smooth

# TODO: Check which subset is the best for hospitalization data


filepath = f"./data/covid_data/covid-hospitalization-all-state-merged_vEW{epiweek}.csv"
df = pd.read_csv(filepath)
df = df[df["region"] == region]

df = df[include_cols_flu]

# Fill missing data
df = df.fillna(method="ffill")
df = df.fillna(method="bfill")
df = df.fillna(0)

# Average smoothing
if smooth == 1:
    df = df.rolling(7, 1).mean()

# Convert to numpy array
df = df.to_numpy()

# Normalize


# Save to pickle
os.makedirs("./data/covid_data/saves", exist_ok=True)
with open(f"./data/covid_data/saves/covid_{region}_{epiweek}.pkl", "wb") as f:
    pickle.dump(df, f)


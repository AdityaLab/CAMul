import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


src_file = "./data/demographics/county_statistics.csv"
sum_cols = [
    "county",
    "state",
    "votes16_Donald_Trump",
    "votes16_Hillary_Clinton",
    "votes20_Donald_Trump",
    "votes20_Joe_Biden",
    "total_votes16",
    "total_votes20",
    "TotalPop",
    "Men",
    "Women",
    "Income",
    "Employed",
]

avg_cols = [
    "county",
    "state",
    "Hispanic",
    "White",
    "Black",
    "Native",
    "Asian",
    "Pacific",
    "Unemployment",
    "Drive",
    "Carpool",
    "Transit",
    "Walk",
    "OtherTransp",
    "WorkAtHome",
]

df1 = pd.read_csv(src_file)[sum_cols].fillna(0)
df2 = pd.read_csv(src_file)[avg_cols].fillna(0)

df1 = df1.groupby(["state"]).sum()
df2 = df2.groupby(["state"]).mean()

df = pd.merge(df1, df2, left_on="state", right_on="state")
df['votes16_Donald_Trump']/= df["total_votes16"]
df['votes16_Hillary_Clinton']/= df["total_votes16"]
df['votes20_Donald_Trump']/= df["total_votes20"]
df['votes20_Joe_Biden']/= df["total_votes20"]
df = df.fillna(0)

states = list(df.index)
arr = np.array(df)
arr = StandardScaler().fit_transform(arr)

dc = {s:a for s,a in zip(states, arr)}

with open("./data/demographics/saves/static_demo.pkl", "wb") as fl:
    pickle.dump(dc, fl)

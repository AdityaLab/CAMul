# Create table ... $y_t^i -> e_t^i$
import pandas as pd
from glob import glob
import pickle
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-r", "--region", dest="region", default="X")
(options, args) = parser.parse_args()

region = options.region

include_cols = [
    "retail_and_recreation_percent_change_from_baseline",
    "grocery_and_pharmacy_percent_change_from_baseline",
    "parks_percent_change_from_baseline",
    "transit_stations_percent_change_from_baseline",
    "workplaces_percent_change_from_baseline",
    "residential_percent_change_from_baseline",
    "covidnet",
    "positiveIncrease",
    "negativeIncrease",
    "totalTestResultsIncrease",
    "onVentilatorCurrently",
    "inIcuCurrently",
    "recovered",
    "hospitalizedIncrease",
    "death_jhu_incidence",
    "dex_a",
    "apple_mobility",
    "Number of Facilities Reporting",
    "CLI Percent of Total Visits",
]

files = sorted(glob("./data/covid_data/covid-hospitalization-all-state-merged*.csv"))
file_to_week = lambda st: int(st[-6:-4]) + int(st[-7])*53
# :3 to avoid missing features
weeks = [file_to_week(f) for f in files][3:]
dfs = [pd.read_csv(f) for f in files][3:]
print()
for i in range(len(weeks)):
    print(
        f"{weeks[i]}, {len(set(include_cols).intersection(dfs[i].columns))}, {dfs[i]['region'].unique().shape}"
    )

start_week = 21
end_week = weeks[-1] - 1

regions = [options.region]
new_dfs = []
for df in dfs:
    df = df.fillna(method="ffill")
    df = df.fillna(method="backfill")
    df = df.fillna(0)
    df["epiweek"] = df["epiweek"] % 100 + 53*((df["epiweek"]//100)%10)
    df = df[(df["epiweek"] >= start_week)]
    df = df[["epiweek", "region"] + include_cols]
    df = df[df["region"].isin(regions)]
    new_dfs.append(df)

dfs = new_dfs

dt = dict()
diff_dt = dict()

for feat in include_cols:
    diff_feat = {}
    feat_vals = {}
    # Get backfill values for each epiweek
    for w in range(start_week, end_week + 1):
        wk_vals = []
        for df, wk in zip(dfs, weeks):
            if wk < w:
                continue
            wk_vals.append(
                float(df[(df["epiweek"] == w) & (df["region"] == regions[0])][feat])
            )
        diff_vals = [wk_vals[-1] - wkv for wkv in wk_vals]
        feat_vals[w] = wk_vals
        diff_feat[w] = diff_vals
    dt[feat] = feat_vals
    diff_dt[feat] = diff_feat

with open(f"./data/covid_data/saves/backfill_vals_{regions[0]}.pkl", "wb") as f:
    pickle.dump((dt, diff_dt), f)

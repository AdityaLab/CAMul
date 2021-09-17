import numpy as np
import pickle
import os

regions = ["AR", "CA", "FL", "GA", "IL", "MA", "NJ", "NY", "OH", "TX", "X"]
features = [
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

def get_bf_file(state):
    filename = f"./saves/backfill_vals_{state}.pkl"
    assert os.path.exists(filename)
    with open(filename, "rb") as fl:
        ans = pickle.load(fl)[0]
    return ans

def get_vector(epiweek, fillweek, region, features=features):
    assert fillweek>=epiweek
    data = get_bf_file(region)
    ans = []
    for f in features:
        ans.append(data[f][epiweek][fillweek-epiweek])
    return np.array(ans)


def stack_history(epiweek, region, last=None, max_fillweek=48):
    assert epiweek<=max_fillweek
    ans = []
    if last is not None:
        start_week = max_fillweek - last
        assert start_week >= epiweek
    else:
        start_week = epiweek
    for w in range(start_week, max_fillweek):
        ans.append(get_vector(epiweek, w, region))
    return np.array(ans)
    
# Schema of saved files

1. The file names are `backfill_vals_<state>.pkl` stored in `./data/covid_data/saves`
2. Each file has two objects: a. dt, b. diff_dt
3. dt is the main file
4. dt is a dictionary of features listed in `./covid_extract/create_table.py`'s `include_cols`
5. Each entry is inturn a dictionary with keys as epiweek numbers followed and contains revision history for the given observed week
6. So use `dt[feat][week][-1]` for the latest version of the signal `feat` fr given week.
7. *diff_dt* is same as *dt* except it contains difference w.r.t final revised value: $diff\_dt[feat][week][t] = dt[feat][week][-1] - dt[feat][week][t]$
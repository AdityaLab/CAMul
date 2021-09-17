# CaMUL: Calibrated and Accurate Multi-view Time-Series Forecasting


## Setup
We require you to have `anaconda` or `miniconda` installed. Run the script `./scripts/setup.sh` to setup the virtual environment with all the required packages.

## Data download and preprocessing

### Twitter dataset

The probability distributions for each week over all states are available in `./data/tweet_dataset` folder as npy files for each week and state.

### Power dataset

Run the `./scripts/download_power.sh` to download dataset.

### Covid dataset

Covid dataset is available in `./data/covid_data` folder. Run `./scripts/covid_preprocess.sh` to preprocess the features of dataset.

### Google Symptoms

Symptoms dataset is available in `./data/symptom_data`. Run `./scripts/preprocess_symp.sh` for preprocessing.

## Experiments

We have `./train_tweets.py`, `./train_covid.py`, `./train_power.py`, `./train_symp.py` to run the model for each of the benchmarks. You may tune the arguments related week ahead, prediction week/season by passing the commandline arguments. Use the `--help` flag for a list of all arguments.



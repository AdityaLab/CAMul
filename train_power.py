import pickle
from optparse import OptionParser
import networkx as nx

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from symp_extract.consts import include_cols
import dgl
from functools import reduce

from models.utils import device, float_tensor, long_tensor
from models.multimodels import (
    EmbedEncoder,
    GRUEncoder,
    EmbGCNEncoder,
    LatentEncoder,
    CorrEncoder,
    Decoder,
)


parser = OptionParser()
parser.add_option("-w", "--week", dest="week_ahead", type="int", default=2)
parser.add_option("-y", "--year", dest="year", type="int", default=2020)
parser.add_option("-n", "--num", dest="num", type="string")
parser.add_option("-e", "--epoch", dest="epochs", type="int", default="1500")
(options, args) = parser.parse_args()

with open("./data/household_power_consumption/household_power_consumption.txt", "r") as f:
    data = f.readlines()

data = [d.strip().split(";") for d in data][1:]

def get_month(ss: str):
    i = ss.find("/")
    return int(ss[i+1:ss[i+1:].find("/")+i + 1]) - 1
def get_time_of_day(ss: str):
    hour = int(ss[:2])
    if hour < 6:
        return 0
    elif hour < 12:
        return 1
    elif hour < 18:
        return 2
    else:
        return 3

tod = np.array([get_time_of_day(d[1]) for d in data], dtype=np.int32)
month = np.array([get_month(d[0]) for d in data], dtype=np.int32)
features = []
for d in data:
    f = []
    for x in d[2:]:
        try:
            f.append(float(x))
        except:
            f.append(0.0)
    features.append(f)
features = np.array(features)
target = features[:, 0]

total_time = len(data)
test_start = int(total_time * 0.8)

X, X_symp, Y, mt, reg = [], [], [], [], []

def sample_train(n_samples, window = 20):
    X, X_symp, Y, mt, reg = [], [], [], [], []
    start_seqs = np.random.randint(0, test_start, n_samples)
    for start_seq in start_seqs:
        X.append(target[start_seq:start_seq+window, np.newaxis])
        X_symp.append(features[start_seq:start_seq+window])
        Y.append(target[start_seq+window])
        mt.append(month[start_seq+window])
        reg.append(tod[start_seq+window])
    X = np.array(X)
    X_symp = np.array(X_symp)
    Y = np.array(Y)
    mt = np.array(mt)
    reg = np.array(reg)
    return X, X_symp, Y, mt, reg

def sample_test(n_samples, window = 20):
    X, X_symp, Y, mt, reg = [], [], [], [], []
    start_seqs = np.random.randint(test_start, total_time, n_samples)
    for start_seq in start_seqs:
        X.append(target[start_seq:start_seq+window, np.newaxis])
        X_symp.append(features[start_seq:start_seq+window])
        Y.append(target[start_seq+window])
        mt.append(month[start_seq+window])
        reg.append(tod[start_seq+window])
    X = np.array(X)
    X_symp = np.array(X_symp)
    Y = np.array(Y)
    mt = np.array(mt)
    reg = np.array(reg)
    return X, X_symp, Y, mt, reg




# Reference points
splits = 10
len_seq = test_start//splits
seq_references = np.array([features[i: i+len_seq, 0, np.newaxis] for i in range(0, test_start, len_seq)])[:, :100, :]
symp_references = np.array([features[i: i+len_seq] for i in range(0, test_start, len_seq)])[:, :100, :]
month_references = np.arange(12)
reg_references = np.arange(4)

train_seqs, train_symp_seqs, train_y, mt, reg = sample_train(100)




month_enc = EmbedEncoder(in_size=12, out_dim=60).to(device)
seq_encoder = GRUEncoder(in_size=1, out_dim=60).to(device)
symp_encoder = GRUEncoder(in_size=7, out_dim=60).to(device)
reg_encoder = EmbedEncoder(in_size=5, out_dim=60).to(device)

stoch_month_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
stoch_seq_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
stoch_symp_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
stoch_reg_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)

month_corr = CorrEncoder(
    in_data_dim=60,
    in_data_det_dim=60,
    in_ref_dim=60,
    in_ref_det_dim=60,
    hidden_dim=60,
    q_layers=2,
    same_decoder=True,
).to(device)
seq_corr = CorrEncoder(
    in_data_dim=60,
    in_data_det_dim=60,
    in_ref_dim=60,
    in_ref_det_dim=60,
    hidden_dim=60,
    q_layers=2,
    same_decoder=True,
).to(device)
symp_corr = CorrEncoder(
    in_data_dim=60,
    in_data_det_dim=60,
    in_ref_dim=60,
    in_ref_det_dim=60,
    hidden_dim=60,
    q_layers=2,
    same_decoder=True,
).to(device)
reg_corr = CorrEncoder(
    in_data_dim=60,
    in_data_det_dim=60,
    in_ref_dim=60,
    in_ref_det_dim=60,
    hidden_dim=60,
    q_layers=2,
    same_decoder=True,
).to(device)

decoder = Decoder(z_dim=60, sr_dim=60, latent_dim=60, hidden_dim=60, y_dim=1).to(device)

models = [
    month_enc,
    seq_encoder,
    symp_encoder,
    reg_encoder,
    stoch_month_enc,
    stoch_seq_enc,
    stoch_symp_enc,
    stoch_reg_enc,
    month_corr,
    seq_corr,
    symp_corr,
    reg_corr,
    decoder,
]

opt = optim.Adam(
    reduce(lambda x, y: x + y, [list(m.parameters()) for m in models]), lr=1e-3
)

# Porbabilistic encode of reference points
ref_months = month_enc.forward(long_tensor(month_references))
ref_seq = seq_encoder.forward(float_tensor(seq_references))
ref_symp = symp_encoder.forward(float_tensor(symp_references))
ref_reg = reg_encoder.forward(long_tensor(reg_references))

stoch_ref_months = stoch_month_enc.forward(ref_months)[0]
stoch_ref_seq = stoch_seq_enc.forward(ref_seq)[0]
stoch_ref_symp = stoch_symp_enc.forward(ref_symp)[0]
stoch_ref_reg = stoch_reg_enc.forward(ref_reg)[0]

# Probabilistic encode of training points

train_months = month_enc.forward(long_tensor(mt.astype(int)))
train_seq = seq_encoder.forward(float_tensor(train_seqs))
train_symp = symp_encoder.forward(float_tensor(train_symp_seqs))
train_reg = reg_encoder.forward(long_tensor(reg.astype(int)))

stoch_train_months = stoch_month_enc.forward(train_months)[0]
stoch_train_seq = stoch_seq_enc.forward(train_seq)[0]
stoch_train_symp = stoch_symp_enc.forward(train_symp)[0]
stoch_train_reg = stoch_reg_enc.forward(train_reg)[0]


def train(train_seqs, train_symp_seqs, reg, mt, train_y):
    for m in models:
        m.train()
    opt.zero_grad()

    # Porbabilistic encode of reference points
    ref_months = month_enc.forward(long_tensor(month_references))
    ref_seq = seq_encoder.forward(float_tensor(seq_references))
    ref_symp = symp_encoder.forward(float_tensor(symp_references))
    ref_reg = reg_encoder.forward(long_tensor(reg_references))

    stoch_ref_months = stoch_month_enc.forward(ref_months)[0]
    stoch_ref_seq = stoch_seq_enc.forward(ref_seq)[0]
    stoch_ref_symp = stoch_symp_enc.forward(ref_symp)[0]
    stoch_ref_reg = stoch_reg_enc.forward(ref_reg)[0]

    # Probabilistic encode of training points

    train_months = month_enc.forward(long_tensor(mt.astype(int)))
    train_seq = seq_encoder.forward(float_tensor(train_seqs))
    train_symp = symp_encoder.forward(float_tensor(train_symp_seqs))
    train_reg = reg_encoder.forward(long_tensor(reg.astype(int)))

    stoch_train_months = stoch_month_enc.forward(train_months)[0]
    stoch_train_seq = stoch_seq_enc.forward(train_seq)[0]
    stoch_train_symp = stoch_symp_enc.forward(train_symp)[0]
    stoch_train_reg = stoch_reg_enc.forward(train_reg)[0]
    # Get view-aware latent embeddings
    train_months_z, train_month_sr, _, month_loss, _ = month_corr.forward(
        stoch_ref_months, stoch_train_months, ref_months, train_months
    )
    train_seq_z, train_seq_sr, _, seq_loss, _ = seq_corr.forward(
        stoch_ref_seq, stoch_train_seq, ref_seq, train_seq
    )
    train_symp_z, train_symp_sr, _, symp_loss, _ = symp_corr.forward(
        stoch_ref_symp, stoch_train_symp, ref_symp, train_symp
    )
    train_reg_z, train_reg_sr, _, reg_loss, _ = reg_corr.forward(
        stoch_ref_reg, stoch_train_reg, ref_reg, train_reg
    )

    # Concat all latent embeddings
    train_z = torch.stack(
        [train_months_z, train_seq_z, train_symp_z, train_reg_z], dim=1
    )
    train_sr = torch.stack(
        [train_month_sr, train_seq_sr, train_symp_sr, train_reg_sr], dim=1
    )

    loss, mean_y, _, _ = decoder.forward(
        train_z, train_sr, train_seq, float_tensor(train_y)[:, None]
    )

    losses = month_loss + seq_loss + symp_loss + reg_loss + loss
    losses.backward()
    opt.step()
    print(f"Loss = {loss.detach().cpu().numpy()}")

    return (
        mean_y.detach().cpu().numpy(),
        losses.detach().cpu().numpy(),
        loss.detach().cpu().numpy(),
    )


def evaluate(test_seqs, test_symp_seqs, reg_test, mt_test, test_y, sample=True):
    for m in models:
        m.eval()
    # Porbabilistic encode of reference points
    ref_months = month_enc.forward(long_tensor(month_references))
    ref_seq = seq_encoder.forward(float_tensor(seq_references))
    ref_symp = symp_encoder.forward(float_tensor(symp_references))
    ref_reg = reg_encoder.forward(long_tensor(reg_references))

    stoch_ref_months = stoch_month_enc.forward(ref_months)[0]
    stoch_ref_seq = stoch_seq_enc.forward(ref_seq)[0]
    stoch_ref_symp = stoch_symp_enc.forward(ref_symp)[0]
    stoch_ref_reg = stoch_reg_enc.forward(ref_reg)[0]

    # Probabilistic encode of test points

    test_months = month_enc.forward(long_tensor(mt_test.astype(int)))
    test_seq = seq_encoder.forward(float_tensor(test_seqs))
    test_symp = symp_encoder.forward(float_tensor(test_symp_seqs))
    test_reg = reg_encoder.forward(long_tensor(reg_test.astype(int)))
    stoch_test_months = stoch_month_enc.forward(test_months)[0]
    stoch_test_seq = stoch_seq_enc.forward(test_seq)[0]
    stoch_test_symp = stoch_symp_enc.forward(test_symp)[0]
    stoch_test_reg = stoch_reg_enc.forward(test_reg)[0]
    # Get view-aware latent embeddings
    test_months_z, test_month_sr, _, _, _, _ = month_corr.predict(
        stoch_ref_months, stoch_test_months, ref_months, test_months
    )
    test_seq_z, test_seq_sr, _, _, _, _ = seq_corr.predict(
        stoch_ref_seq, stoch_test_seq, ref_seq, test_seq
    )
    test_symp_z, test_symp_sr, _, _, _, _ = symp_corr.predict(
        stoch_ref_symp, stoch_test_symp, ref_symp, test_symp
    )
    test_reg_z, test_reg_sr, _, _, _, _ = reg_corr.predict(
        stoch_ref_reg, stoch_test_reg, ref_reg, test_reg
    )

    # Concat all latent embeddings
    test_z = torch.stack([test_months_z, test_seq_z, test_symp_z, test_reg_z], dim=1)
    test_sr = torch.stack(
        [test_month_sr, test_seq_sr, test_symp_sr, test_reg_sr], dim=1
    )

    sample_y, mean_y, _, _ = decoder.predict(
        test_z, test_sr, test_seq, sample=sample
    )
    sample_y = sample_y.detach().cpu().numpy().ravel()
    mean_y = mean_y.detach().cpu().numpy().ravel()

    # RMSE loss
    rmse = np.sqrt(np.mean((sample_y - test_y.ravel()) ** 2))
    # Mean absolute error
    # mae = np.mean(np.abs(sample_y - test_y))

    print(f"RMSE = {rmse}")
    return rmse, mean_y, sample_y


for ep in range(1, 1000 + 1):
    train_seqs, train_symp_seqs, train_y, mt, reg = sample_train(100)
    train(train_seqs, train_symp_seqs, reg, mt, train_y)
    if ep % 10 == 0:
        print("Evaluating")
        test_seqs, test_symp_seqs, test_y, mt_test, reg_test = sample_test(100)
        evaluate(test_seqs, test_symp_seqs, reg_test, mt_test, test_y)

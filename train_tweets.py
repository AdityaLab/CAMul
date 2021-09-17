from os import stat
import pickle
from optparse import OptionParser
import networkx as nx

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import full
import torch
import torch.optim as optim
import torch.nn as nn
import dgl
from functools import reduce

from models.utils import device, float_tensor, long_tensor
from models.multimodels import (
    EmbedEncoder,
    GRUEncoder,
    FFN,
    EmbGCNEncoder,
    LatentEncoder,
    CorrEncoder,
    Decoder,
)


parser = OptionParser()
parser.add_option("-w", "--week", dest="week_ahead", type="int", default=2)
parser.add_option("-p", "--pred", dest="week_pred", type="int", default=18)
parser.add_option("-y", "--year", dest="year", type="int", default=2020)
parser.add_option("-n", "--num", dest="num", type="string")
parser.add_option("-e", "--epoch", dest="epochs", type="int", default="1500")
(options, args) = parser.parse_args()
topic = 0

week_ahead = options.week_ahead
pred_week = options.week_pred
num = options.num
epochs = options.epochs
year = options.year

# Extract dataset
states = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "DC",
    "FL",
    "GA",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]
state_idx = {s: i for i, s in enumerate(states)}
start_week = 0
train_weeks = np.arange(start_week, pred_week - week_ahead + 1)

def get_week(week, region):
    file = f"./data/tweet_dataset/week{week}_{region}.npy"
    return np.load(file)
 
full_sequences = np.array([np.array([get_week(w, r) for w in np.arange(start_week, pred_week + 1)]) for r in states])

with open("./data/demographics/saves/adj_list.pkl", "rb") as f:
    adj_list = pickle.load(f)
graph = nx.Graph()
graph.add_nodes_from(list(range(len(states))))
for s in states:
    graph.add_edges_from([(state_idx[s], state_idx[d]) for d in adj_list[s]])

graph = dgl.from_networkx(graph)
graph = dgl.add_self_loop(graph)

with open("./data/demographics/saves/static_demo.pkl", "rb") as f:
    static_demo = pickle.load(f)

static_demo = np.array([static_demo[s] for s in states])

def week_to_month(week: int):
    return (week) // 4 + 1





def seq_to_dataset(seq,  state, week_ahead=week_ahead):
    X, Y, wk, reg, static = [], [], [], [], []
    start_idx = max(week_ahead, seq.shape[0] - 32 + week_ahead)
    for i in range(start_idx, seq.shape[0]):
        X.append(seq[: i - week_ahead + 1])
        Y.append(seq[i])
        wk.append(week_to_month(i))
        reg.append(state)
        static.append(static_demo[state])
    return X, Y, wk, reg, static

X, X_symp, Y, mt, reg, static = [], [], [], [], [], []
for i, s in enumerate(states):
    ans = seq_to_dataset(full_sequences[i, :train_weeks[-1]], state_idx[s])
    X.extend([x[:, topic] for x in ans[0]])
    X_symp.extend([x for x in ans[0]])
    Y.extend([x[topic] for x in ans[1]])
    mt.extend(ans[2])
    reg.extend(ans[3])
    static.extend(ans[4])

X_test, X_symp_test, Y_test, mt_test, reg_test, static_test = [], [], [], [], [], []
for i, s in enumerate(states):
    ans = seq_to_dataset(full_sequences[i], state_idx[s], pred_week)
    X_test.append(full_sequences[i, :train_weeks[-1], topic])
    X_symp_test.append(full_sequences[i, :train_weeks[-1]])
    Y_test.append(full_sequences[i, -1, topic])
    mt_test.append(week_to_month(pred_week))
    reg_test.append(i)
    static_test.append(static_demo[i])


# Reference points
seq_references = full_sequences[:,:,topic]
symp_references = full_sequences
month_references = np.arange(12)
reg_references = np.arange(len(states))
static_reference = static_demo




def preprocess_seq_batch(seq_list: list):
    max_len = max([len(x) for x in seq_list])
    if len(seq_list[0].shape) == 2:
        ans = np.zeros((len(seq_list), max_len, len(seq_list[0][0])))
    else:
        ans = np.zeros((len(seq_list), max_len, 1))
        seq_list = [x[:, np.newaxis] for x in seq_list]
    for i, seq in enumerate(seq_list):
        ans[i, : len(seq), :] = seq
    return ans


seq_references = preprocess_seq_batch(seq_references)
symp_references = preprocess_seq_batch(symp_references)


train_seqs = preprocess_seq_batch(X)
train_y = np.array(Y)
train_symp_seqs = preprocess_seq_batch(X_symp)
mt = np.array(mt, dtype=np.int32)
mt_test = np.array(mt_test, dtype=np.int32)
reg = np.array(reg, dtype=np.int32)
reg_test = np.array(reg_test, dtype=np.int32)
static = np.array(static, dtype=np.float32)
static_test = np.array(static_test, dtype=np.float32)
test_seqs = preprocess_seq_batch(X_test)
test_symp_seqs = preprocess_seq_batch(X_symp_test)
test_y = np.array(Y_test)


month_enc = EmbedEncoder(in_size=12, out_dim=60).to(device)
seq_encoder = GRUEncoder(in_size=1, out_dim=60).to(device)
symp_encoder = GRUEncoder(in_size=30, out_dim=60).to(device)
static_encoder = FFN(
    in_dim=24, hidden_layers=[60, 60], out_dim=60
).to(device)
reg_encoder = EmbGCNEncoder(
    in_size=50, emb_dim=60, out_dim=60, num_layers=2, device=device
).to(device)

stoch_month_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
stoch_seq_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
stoch_symp_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
stoch_reg_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
stoch_static_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)

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
static_corr = CorrEncoder(
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
    static_encoder,
    stoch_month_enc,
    stoch_seq_enc,
    stoch_symp_enc,
    stoch_reg_enc,
    stoch_static_enc,
    month_corr,
    seq_corr,
    symp_corr,
    reg_corr,
    static_corr,
    decoder,
]

opt = optim.Adam(
    reduce(lambda x, y: x + y, [list(m.parameters()) for m in models]), lr=1e-3
)


def train(train_seqs, train_symp_seqs, reg, mt, train_y):
    for m in models:
        m.train()
    opt.zero_grad()

    # Porbabilistic encode of reference points
    ref_months = month_enc.forward(long_tensor(month_references))
    ref_seq = seq_encoder.forward(float_tensor(seq_references))
    ref_symp = symp_encoder.forward(float_tensor(symp_references))
    ref_reg = reg_encoder.forward(long_tensor(reg_references), graph.to(device))
    ref_static = static_encoder.forward(float_tensor(static_reference))

    stoch_ref_months = stoch_month_enc.forward(ref_months)[0]
    stoch_ref_seq = stoch_seq_enc.forward(ref_seq)[0]
    stoch_ref_symp = stoch_symp_enc.forward(ref_symp)[0]
    stoch_ref_reg = stoch_reg_enc.forward(ref_reg)[0]
    stoch_ref_static = stoch_static_enc.forward(ref_static)[0]

    # Probabilistic encode of training points

    train_months = month_enc.forward(long_tensor(mt.astype(int)))
    train_seq = seq_encoder.forward(float_tensor(train_seqs))
    train_symp = symp_encoder.forward(float_tensor(train_symp_seqs))
    train_reg = torch.stack([ref_reg[i] for i in reg], dim=0)
    train_static = static_encoder.forward(float_tensor(train_reg))

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
    ref_reg = reg_encoder.forward(long_tensor(reg_references), graph.to(device))

    stoch_ref_months = stoch_month_enc.forward(ref_months)[0]
    stoch_ref_seq = stoch_seq_enc.forward(ref_seq)[0]
    stoch_ref_symp = stoch_symp_enc.forward(ref_symp)[0]
    stoch_ref_reg = stoch_reg_enc.forward(ref_reg)[0]

    # Probabilistic encode of test points

    test_months = month_enc.forward(long_tensor(mt_test.astype(int)))
    test_seq = seq_encoder.forward(float_tensor(test_seqs))
    test_symp = symp_encoder.forward(float_tensor(test_symp_seqs))
    test_reg = torch.stack([ref_reg[i] for i in reg_test], dim=0)

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

    sample_y, mean_y, _, _ = decoder.predict(test_z, test_sr, test_seq, sample=sample)
    sample_y = sample_y.detach().cpu().numpy().ravel()
    mean_y = mean_y.detach().cpu().numpy().ravel()

    # RMSE loss
    rmse = np.sqrt(np.mean((sample_y - test_y.ravel()) ** 2))
    # Mean absolute error
    # mae = np.mean(np.abs(sample_y - test_y))

    print(f"RMSE = {rmse}")
    return rmse, mean_y, sample_y


for ep in range(1, epochs + 1):
    train(train_seqs, train_symp_seqs, reg, mt, train_y)
    if ep % 10 == 0:
        print("Evaluating")
        evaluate(test_seqs, test_symp_seqs, reg_test, mt_test, test_y)

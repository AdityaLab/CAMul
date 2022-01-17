import numpy as np
import pickle
import os

from optparse import OptionParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from covid_extract.hosp_consts import include_cols

from models.multimodels import (
    EmbedEncoder,
    GRUEncoder,
    EmbGCNEncoder,
    LatentEncoder,
    CorrEncoder,
    Decoder,
)
from models.fnpmodels import RegressionFNP2

parser = OptionParser()
parser.add_option("-e", "--epiweek", dest="epiweek", default="202113", type="string")
parser.add_option("--epochs", dest="epochs", default=1500, type="int")
parser.add_option("--lr", dest="lr", default=1e-3, type="float")
parser.add_option("--patience", dest="patience", default=100, type="int")
parser.add_option("--tolerance", dest="tol", default=0.1, type="float")
parser.add_option("-w", "--week", dest="week_ahead", default=1, type="int")
parser.add_option("-s", "--seed", dest="seed", default=0, type="int")
parser.add_option("-b", "--batch", dest="batch_size", default=64, type="int")
parser.add_option("-m", "--save", dest="save_model", default="default", type="string")
parser.add_option("--start_model", dest="start_model", default="None", type="string")
parser.add_option("-c", "--cuda", dest="cuda", default=True, action="store_true")
parser.add_option("--start", dest="start_day", default=-90, type="int")

(options, args) = parser.parse_args()
epiweek = options.epiweek
week_ahead = options.week_ahead
seed = options.seed
save_model_name = options.save_model
start_model = options.start_model
cuda = options.cuda
start_day = options.start_day
batch_size = options.batch_size
lr = options.lr
epochs = options.epochs
patience = options.patience
tol = options.tol

# First do sequence alone
# Then add exo features
# Then TOD (as feature, as view)
# llely demographic features

# Initialize random seed
np.random.seed(seed)
torch.manual_seed(seed)


float_tensor = (
    torch.cuda.FloatTensor
    if (cuda and torch.cuda.is_available())
    else torch.FloatTensor
)
long_tensor = (
    torch.cuda.LongTensor if (cuda and torch.cuda.is_available()) else torch.LongTensor
)
device = "cuda" if (cuda and torch.cuda.is_available()) else "cpu"

# Get dataset as numpy arrays

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
    "X",
]

raw_data = []
for st in states:
    with open(f"./data/covid_data/saves/covid_{st}_{epiweek}.pkl", "rb") as fl:
        raw_data.append(pickle.load(fl))

raw_data = np.array(raw_data)[:, start_day:, :]  # states x days x features
label_idx = include_cols.index("cdc_hospitalized")

raw_data_unnorm = raw_data.copy()


class ScalerFeat:
    def __init__(self, raw_data):
        self.means = np.mean(raw_data, axis=1)
        self.vars = np.std(raw_data, axis=1) + 1e-8

    def transform(self, data):
        return (data - np.transpose(self.means[:, :, None], (0, 2, 1))) / np.transpose(
            self.vars[:, :, None], (0, 2, 1)
        )

    def inverse_transform(self, data):
        return data * np.transpose(self.vars[:, :, None], (0, 2, 1)) + np.transpose(
            self.means[:, :, None], (0, 2, 1)
        )

    def transform_idx(self, data, idx=label_idx):
        return (data - self.means[:, idx]) / self.vars[:, idx]

    def inverse_transform_idx(self, data, idx=label_idx):
        return data * self.vars[:, idx] + self.means[:, idx]


scaler = ScalerFeat(raw_data)
raw_data = scaler.transform(raw_data)


# Chunk dataset sequences


def prefix_sequences(seq, week_ahead=week_ahead):
    """
    Prefix sequences with zeros
    """
    l = len(seq)
    X, Y = np.zeros((l - week_ahead, l, seq.shape[-1])), np.zeros(l - week_ahead)
    for i in range(l - week_ahead):
        X[i, (l - i - 1) :, :] = seq[: i + 1, :]
        Y[i] = seq[i + week_ahead, label_idx]
    return X, Y


X, Y = [], []
for i, st in enumerate(states):
    x, y = prefix_sequences(raw_data[i])
    X.append(x)
    Y.append(y)

X_train, Y_train = np.concatenate(X), np.concatenate(Y)

# Shuffle data
perm = np.random.permutation(len(X_train))
X_train, Y_train = X_train[perm], Y_train[perm]

# Reference sequences
X_ref = raw_data[:, :, label_idx]

# Divide val and train
frac = 0.1
X_val, Y_val = X_train[: int(len(X_train) * frac)], Y_train[: int(len(X_train) * frac)]
X_train, Y_train = (
    X_train[int(len(X_train) * frac) :],
    Y_train[int(len(X_train) * frac) :],
)


# Build model
feat_enc = GRUEncoder(in_size=len(include_cols), out_dim=60, bidirectional=True).to(
    device
)
seq_enc = GRUEncoder(in_size=1, out_dim=60, bidirectional=True).to(device)
fnp_enc = RegressionFNP2(
    dim_x=60,
    dim_y=1,
    dim_h=100,
    n_layers=3,
    num_M=batch_size,
    dim_u=60,
    dim_z=60,
    use_DAG=False,
    use_ref_labels=False,
    add_atten=False,
).to(device)


def load_model(folder, file=save_model_name):
    """
    Load model
    """
    full_path = os.path.join(folder, file)
    assert os.path.exists(full_path)
    feat_enc.load_state_dict(torch.load(os.path.join(full_path, "feat_enc.pt")))
    seq_enc.load_state_dict(torch.load(os.path.join(full_path, "seq_enc.pt")))
    fnp_enc.load_state_dict(torch.load(os.path.join(full_path, "fnp_enc.pt")))


def save_model(folder, file=save_model_name):
    """
    Save model
    """
    full_path = os.path.join(folder, file)
    os.makedirs(full_path, exist_ok=True)
    torch.save(feat_enc.state_dict(), os.path.join(full_path, "feat_enc.pt"))
    torch.save(seq_enc.state_dict(), os.path.join(full_path, "seq_enc.pt"))
    torch.save(fnp_enc.state_dict(), os.path.join(full_path, "fnp_enc.pt"))


# Build dataset
class SeqData(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y[:, None]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            float_tensor(self.X[idx, :, :]),
            float_tensor(self.Y[idx]),
        )


train_dataset = SeqData(X_train, Y_train)
val_dataset = SeqData(X_val, Y_val)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True
)

if start_model != "None":
    load_model("./mort_models", file=start_model)
    print("Loaded model from", start_model)

opt = torch.optim.Adam(
    list(seq_enc.parameters())
    + list(feat_enc.parameters())
    + list(fnp_enc.parameters()),
    lr=lr,
)


def train_step(data_loader, X, Y, X_ref):
    """
    Train step
    """
    feat_enc.train()
    seq_enc.train()
    fnp_enc.train()
    total_loss = 0.0
    train_err = 0.0
    YP = []
    T_target = []
    for i, (x, y) in enumerate(data_loader):
        opt.zero_grad()
        x_seq = seq_enc(float_tensor(X_ref).unsqueeze(2))
        x_feat = feat_enc(x)
        loss, yp, _ = fnp_enc(x_seq, float_tensor(X_ref), x_feat, y)
        yp = yp[X_ref.shape[0] :]
        loss.backward()
        opt.step()
        YP.append(yp.detach().cpu().numpy())
        T_target.append(y.detach().cpu().numpy())
        total_loss += loss.detach().cpu().numpy()
        train_err += torch.pow(yp - y, 2).mean().sqrt().detach().cpu().numpy()
    return (
        total_loss / (i + 1),
        train_err / (i + 1),
        np.array(YP).ravel(),
        np.array(T_target).ravel(),
    )


def val_step(data_loader, X, Y, X_ref, sample=True):
    """
    Validation step
    """
    with torch.set_grad_enabled(False):
        feat_enc.eval()
        seq_enc.eval()
        fnp_enc.eval()
        val_err = 0.0
        YP = []
        T_target = []
        for i, (x, y) in enumerate(data_loader):
            x_seq = seq_enc(float_tensor(X_ref).unsqueeze(2))
            x_feat = feat_enc(x)
            yp, _, vars, _, _, _, _ = fnp_enc.predict(
                x_feat, x_seq, float_tensor(X_ref), sample
            )
            val_err += torch.pow(yp - y, 2).mean().sqrt().detach().cpu().numpy()
            YP.append(yp.detach().cpu().numpy())
            T_target.append(y.detach().cpu().numpy())
        return val_err / (i + 1), np.array(YP).ravel(), np.array(T_target).ravel()


def test_step(X, X_ref, samples=1000):
    """
    Test step
    """
    with torch.set_grad_enabled(False):
        feat_enc.eval()
        seq_enc.eval()
        fnp_enc.eval()
        YP = []
        for i in range(samples):
            x_seq = seq_enc(float_tensor(X_ref).unsqueeze(2))
            x_feat = feat_enc(float_tensor(X))
            yp, _, vars, _, _, _, _ = fnp_enc.predict(
                x_feat, x_seq, float_tensor(X_ref), sample=True
            )
            YP.append(yp.detach().cpu().numpy())
        return np.array(YP)


min_val_err = np.inf
min_val_epoch = 0
all_losses = []
for ep in range(epochs):
    print(f"Epoch {ep+1}")
    train_loss, train_err, yp, yt = train_step(train_loader, X_train, Y_train, X_ref)
    print(f"Train loss: {train_loss:.4f}, Train err: {train_err:.4f}")
    val_err, yp, yt = val_step(val_loader, X_val, Y_val, X_ref)
    print(f"Val err: {val_err:.4f}")
    all_losses.append(val_err)
    if val_err < min_val_err:
        min_val_err = val_err
        min_val_epoch = ep
        save_model("./mort_models")
        print("Saved model")
    print()
    print()
    if (
        ep > 100
        and ep - min_val_epoch > patience
        and min(all_losses[-patience:]) > min_val_err(1.0 + tol)
    ):
        break

# Now we get results
load_model("./mort_models")
X_test = raw_data
Y_test = test_step(X_test, X_ref, samples=2000).squeeze()

Y_test_unnorm = scaler.inverse_transform_idx(Y_test, label_idx)

# Save predictions
os.makedirs(f"./mort_predictions", exist_ok=True)
with open(f"./mort_predictions/{save_model_name}_predictions.pkl", "wb") as f:
    pickle.dump(Y_test_unnorm, f)

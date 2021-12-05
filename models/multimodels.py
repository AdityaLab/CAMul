import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F
from torch.nn.modules import activation
from models.utils import (
    Normal,
    float_tensor,
    logitexp,
    sample_DAG,
    sample_Clique,
    sample_bipartite,
    Flatten,
    one_hot,
    device,
)
from torch.distributions import Categorical
from typing import List
import dgl
import dgl.nn as dglnn


class FFN(nn.Module):
    """
    Generic Feed Forward Network class
    """

    def __init__(
        self,
        in_dim: int,
        hidden_layers: List[int],
        out_dim: int,
        activation=nn.ReLU,
        dropout: float = 0.0,
    ):
        r"""
        ## Inputs

        :param in_dim: Input dimensions
        :param hidden_layers: List of hidden layer sizes
        :param out_dim: Output dimensions
        :param activation: nn Module for activation
        :param Dropout: rate of dropout
        ```math
        e=mc^2
        ```
        """
        super(FFN, self).__init__()
        self.layers = [nn.Linear(in_dim, hidden_layers[0]), activation()]
        for i in range(1, len(hidden_layers)):
            self.layers.extend(
                [
                    nn.Linear(hidden_layers[i - 1], hidden_layers[i]),
                    activation(),
                    nn.Dropout(dropout),
                ]
            )
        self.layers.append(nn.Linear(hidden_layers[-1], out_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inp):
        r"""
        ## Inputs

        :param inp: Input vectors shape: [batch, inp_dim]

        ----
        ## Outputs

        out: [batch, out_dim]
        """
        return self.layers(inp)


class LatentEncoder(nn.Module):
    """
    Generic Stochastic Encoder using FFN
    """

    def __init__(
        self,
        in_dim: int,
        hidden_layers: List[int],
        out_dim: int,
        activation=nn.ReLU,
        dropout: float = 0.0,
    ):
        r"""
        ## Inputs

        :param in_dim: Input dimensions
        :param hidden_layers: List of hidden layer sizes
        :param out_dim: Output dimensions
        :param activation: nn Module for activation
        :param Dropout: rate of dropout
        """
        super(LatentEncoder, self).__init__()
        self.out_dim = out_dim
        self.net = FFN(in_dim, hidden_layers, out_dim * 2, activation, dropout)

    def forward(self, inp):
        r"""
        ## Inputs

        :param inp: Input vectors shape: [batch, inp_dim]

        ----
        ## Outputs

        mean: [batch, out_dim]
        logscale: [batch, out_dim],
        dist: Normal Distribution with mean and logscale
        """
        out = self.net(inp)
        mean, logscale = torch.split(out, self.out_dim, dim=-1)
        dist = Normal(mean, logscale)
        out = dist.rsample()
        return out, mean, logscale, dist


class EmbedEncoder(nn.Module):
    """
    Encoder for categorical values
    """

    def __init__(self, in_size: int, out_dim: int):
        r"""
        ## Inputs

        :param in_size: Input vocab size
        :param out_dim: Output dimensions
        """
        super(EmbedEncoder, self).__init__()
        self.emb_layer = nn.Embedding(in_size, out_dim)

    def forward(self, batch):
        return self.emb_layer(batch)


class GRUEncoder(nn.Module):
    """
    Encodes Sequences using GRU
    """

    def __init__(self, in_size: int, out_dim: int, bidirectional: bool = False):
        super(GRUEncoder, self).__init__()
        self.out_dim = out_dim // 2 if bidirectional else out_dim
        self.gru = nn.GRU(
            in_size, self.out_dim, batch_first=True, bidirectional=bidirectional
        )

    def forward(self, batch):
        r"""
        ## Inputs

        :param batch: Input vectors shape: [batch, seq_len, in_size]

        ----
        ## Outputs

        out: [batch, seq_len, out_dim]
        """
        out_seq, _ = self.gru(batch)
        return out_seq[:, -1, :]


class EmbGCNEncoder(nn.Module):
    """
    Encoder for categorical values with adj graph
    """

    def __init__(
        self,
        in_size: int,
        emb_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        activation=F.relu,
        device=device,
    ):
        super(EmbGCNEncoder, self).__init__()
        self.device = device
        self.activation = activation
        self.emb_layer = nn.Embedding(in_size, emb_dim)
        self.gcn_layers = [dglnn.GraphConv(emb_dim, out_dim).to(device)]
        for i in range(num_layers - 1):
            self.gcn_layers.append(dglnn.GraphConv(out_dim, out_dim).to(device))

    def forward(self, batch, graph):
        embs = self.emb_layer(batch)
        for i in range(len(self.gcn_layers)):
            embs = self.activation(
                F.dropout(self.gcn_layers[i](graph, embs), training=self.training)
            )
        return embs


class CorrEncoder(nn.Module):
    r"""
    Takes training and reference points and outputs $p(z|x,U)$
    """

    def __init__(
        self,
        in_data_dim: int,
        in_data_det_dim: int,
        in_ref_dim: int,
        in_ref_det_dim: int,
        hidden_dim: int,
        q_layers=1,
        same_decoder=True,
    ):
        r"""
        ## Inputs

        :param in_data_dim: Dimension of input latent x
        :param in_data_det_dim: Dimension of non-stochastic input
        :param in_ref_dim: Dimension of input latent reference x
        :param in_ref_det_dim: Dimension of input non-stochastic reference x
        :param hidden_dim: transformed dimensions for comparing in correlation matrix and output z
        :param q_layers: no. of hidden layers for variational q distribution
        """
        super(CorrEncoder, self).__init__()
        self.in_data_dim = in_data_dim
        self.in_ref_dim = in_ref_dim
        self.hidden_dim = hidden_dim

        self.norm_graph = lambda x: x / (torch.sum(x, 1, keepdim=True) + 1e-8)

        self.data_enc1 = nn.Linear(in_data_det_dim, hidden_dim)
        if same_decoder:
            self.ref_enc1 = self.data_enc1
        else:
            self.ref_enc1 = nn.Linear(in_ref_det_dim, hidden_dim)

        self.data_enc = nn.Linear(in_data_dim, hidden_dim)
        if same_decoder:
            self.ref_enc = self.data_enc
        else:
            self.ref_enc = nn.Linear(in_ref_dim, hidden_dim)
        self.pairwise_g_logscale = nn.Parameter(
            float_tensor(1).fill_(math.log(math.sqrt(self.hidden_dim)))
        )
        self.pairwise_g = lambda x: logitexp(
            -0.5
            * torch.sum(
                torch.pow(x[:, self.hidden_dim :] - x[:, 0 : self.hidden_dim], 2),
                1,
                keepdim=True,
            )
            / self.pairwise_g_logscale.exp()
        ).view(x.size(0), 1)

        self.q_z = FFN(hidden_dim, [hidden_dim] * q_layers, hidden_dim * 2)

    def forward(self, ZR, ZM, XR, XM):
        r"""
        ##Inputs:

        :param ZR: Stochastic reference set
        :param ZM: Stochastic training set
        :param XR: Det. Reference set
        :param XM: Det. training set
        """
        X_all = torch.cat([self.ref_enc1(XR), self.data_enc1(XM)], dim=0)
        HR = self.ref_enc(ZR)
        HM = self.data_enc(ZM)

        A = sample_bipartite(HM, HR, self.pairwise_g, training=self.training)
        qz_mean_all, qz_logscale_all = torch.split(self.q_z(X_all), self.hidden_dim, 1)
        qz = Normal(qz_mean_all, qz_logscale_all)
        z = qz.rsample()
        pz_mean = torch.mm(self.norm_graph(A), qz_mean_all[: XR.size(0)])
        pz_logscale = torch.mm(self.norm_graph(A), qz_logscale_all[: XR.size(0)])
        pz = Normal(pz_mean, pz_logscale)
        pqz = pz.log_prob(z[XR.size(0) :]) - qz.log_prob(z)[XR.size(0) :]
        logstd_pqz_M = torch.sum(pqz[XR.size(0) :])
        z = z[XR.size(0) :]
        final_rep = z

        sR = X_all[: XR.size(0)].mean(dim=0).repeat(final_rep.shape[0], 1)
        final_rep = torch.cat([sR, final_rep], dim=-1)
        # TODO: Fill outputs in docstring
        loss = -(logstd_pqz_M) / float(XM.size(0))
        return z, sR, HM, loss, A

    def predict(self, ZR, ZM, XR, XM, sample=True):
        HR = self.ref_enc(ZR)
        HM = self.data_enc(ZM)

        A = sample_bipartite(HM, HR, self.pairwise_g, training=self.training)
        pz_mean_R, pz_logscale_R = torch.split(
            self.q_z(self.ref_enc1(XR)), self.hidden_dim, 1
        )
        pz_mean_M = torch.mm(self.norm_graph(A), pz_mean_R)
        pz_logscale_M = torch.mm(self.norm_graph(A), pz_logscale_R)
        pz = Normal(pz_mean_M, pz_logscale_M)
        z = pz.rsample()

        final_rep = z

        sR = XR.mean(dim=0).repeat(final_rep.shape[0], 1)
        final_rep = torch.cat([sR, final_rep], dim=-1)

        if sample:
            out = z
        else:
            out = pz_mean_M

        return out, sR, HM, pz_mean_M, pz_logscale_M, A


class MultAttention(nn.Module):
    r"""
    Multiplicative attention similar to Vaswani et al.
    """

    def __init__(self, key_dim: int, val_dim: int, out_dim: int):
        super(MultAttention, self).__init__()
        self.key_encoder = nn.Linear(key_dim, out_dim)
        self.val_encoder = nn.Linear(val_dim, out_dim)
        self.query_encoder = nn.Linear(key_dim, out_dim)

    def forward(self, vals, keys_):
        r"""
        # Inputs:

        :param vals: Values of shape [batch x val_dim]
        :param keys: Keys of shape [batch x graphs x key_dim]
        """
        keys = self.key_encoder(keys_)
        queries = self.query_encoder(keys_)
        vals = self.val_encoder(vals)
        vals = vals.unsqueeze(1)
        weights = torch.matmul(keys, vals.transpose(1, 2))  # [batch x graphs x 1]
        weights = torch.softmax(weights, 1)
        summed_queries = (queries * weights).sum(1)
        return summed_queries, weights


class TanhAttention(nn.Module):
    r"""
    Tanh Attention
    """

    def __init__(self, key_dim: int, val_dim: int, out_dim: int):
        super(TanhAttention, self).__init__()
        self.key_encoder = nn.Linear(key_dim, out_dim)
        self.val_encoder = nn.Linear(val_dim, out_dim)
        self.query_encoder = nn.Linear(key_dim, out_dim)

    def forward(self, vals, keys_):
        r"""
        # Inputs:

        :param vals: Values of shape [batch x val_dim]
        :param keys: Keys of shape [batch x graphs x key_dim]
        """
        # TODO: Implement formula
        keys = self.key_encoder(keys_)
        queries = self.query_encoder(keys_)
        vals = self.val_encoder(vals)
        vals = vals.unsqueeze(1)
        weights = torch.matmul(keys, vals.transpose(1, 2))  # [batch x graphs x 1]
        weights = torch.softmax(weights, 1)
        summed_queries = (queries * weights).sum(1)
        return summed_queries, weights


class Decoder(nn.Module):
    r"""
    takes z from all decoders and outputs $p(y)$
    """

    def __init__(
        self, z_dim: int, sr_dim: int, latent_dim: int, hidden_dim: int, y_dim: int
    ):
        r"""
        ## Inputs

        :params z_dim: latent encoding from graphs
        :params sr_dim: global latent encodings from reference sets
        :params latent_dim: latent dims for stochastic encodings
        """
        super(Decoder, self).__init__()
        self.attention_layer = MultAttention(z_dim + sr_dim, latent_dim, hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * y_dim),
        )
        self.y_dim = y_dim

    def forward(self, zs, sRs, HM, y):
        r"""
        ## Inputs

        :params zs: z encoding [batch x graphs x z_dims]
        :params sRs: sR encodings [graphs x sr_dim]
        :params HM: [batch x latent_dim]
        """
        # stack_sRs = torch.stack([sRs]*zs.size(0), 0)
        stack_sRs = sRs
        new_zs = torch.cat([zs, stack_sRs], -1)
        z, attens = self.attention_layer(HM, new_zs)
        mean_y, logstd_y = torch.split(
            self.output(torch.cat([z, HM], -1)), self.y_dim, 1
        )
        py = Normal(mean_y, logstd_y)
        log_pyM = torch.sum(py.log_prob(y))
        loss = -log_pyM / float(zs.size(0))

        return loss, mean_y, logstd_y, attens

    def predict(self, zs, sRs, HM, sample=True):
        r"""
        ## Inputs

        :params zs: z encoding [batch x graphs x z_dims]
        :params sRs: sR encodings [graphs x sr_dim]
        :params HM: [batch x latent_dim]
        """
        # stack_sRs = torch.stack([sRs]*zs.size(0), 0)
        stack_sRs = sRs
        new_zs = torch.cat([zs, stack_sRs], -1)
        z, attens = self.attention_layer(HM, new_zs)
        mean_y, logstd_y = torch.split(
            self.output(torch.cat([z, HM], -1)), self.y_dim, 1
        )
        py = Normal(mean_y, logstd_y)
        if sample:
            out = py.sample()
        else:
            out = mean_y

        return out, mean_y, logstd_y, attens


def test_latent():
    # Define models
    model = LatentEncoder(10, [12, 15], 20).to("cuda")
    inp = torch.rand(50, 10).to("cuda")
    out = model(inp)
    return out


def test_corr():
    model = CorrEncoder(20, 20, 25, 25, 15).to("cuda")
    ref_stoc, ref_det = torch.rand(10, 25).to("cuda"), torch.rand(10, 25).to("cuda")
    inp_stoc, inp_det = torch.rand(50, 20).to("cuda"), torch.rand(50, 20).to("cuda")
    out = model(ref_stoc, inp_stoc, ref_det, inp_det)
    out1 = model.predict(ref_stoc, inp_stoc, ref_det, inp_det)
    return out, out1


def test_decoder():
    model = Decoder(15, 15, 20, 30, 1).to("cuda")
    zs = torch.rand(10, 4, 15).to("cuda")
    sRs = torch.rand(10, 4, 15).to("cuda")
    lat = torch.rand(10, 30).to("cuda")
    y = torch.rand(10, 1).to("cuda")
    out = model(zs, sRs, lat, y)
    out1 = model.predict(zs, sRs, lat)
    return out, out1


if __name__ == "__main__":
    out = test_latent()

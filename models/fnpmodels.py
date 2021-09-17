import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from models.utils import (
    Normal,
    float_tensor,
    logitexp,
    sample_DAG,
    sample_Clique,
    sample_bipartite,
    Flatten,
    one_hot,
)
from torch.distributions import Categorical


class TransformerAttn(nn.Module):
    """
    Module that calculates self-attention weights using transformer like attention
    """

    def __init__(self, dim_in=40, value_dim=40, key_dim=40) -> None:
        """
        param dim_in: Dimensionality of input sequence
        param value_dim: Dimension of value transform
        param key_dim: Dimension of key transform
        """
        super(TransformerAttn, self).__init__()
        self.value_layer = nn.Linear(dim_in, value_dim)
        self.query_layer = nn.Linear(dim_in, value_dim)
        self.key_layer = nn.Linear(dim_in, key_dim)

    def forward(self, seq):
        """
        param seq: Sequence in dimension [Seq len, Batch, Hidden size]
        """
        seq_in = seq.transpose(0, 1)
        value = self.value_layer(seq_in)
        query = self.query_layer(seq_in)
        keys = self.key_layer(seq_in)
        weights = (value @ query.transpose(1, 2)) / math.sqrt(seq.shape[-1])
        weights = torch.softmax(weights, -1)
        return (weights @ keys).transpose(1, 0)

    def forward_mask(self, seq, mask):
        """
        param seq: Sequence in dimension [Seq len, Batch, Hidden size]
        """
        seq_in = seq.transpose(0, 1)
        value = self.value_layer(seq_in)
        query = self.query_layer(seq_in)
        keys = self.key_layer(seq_in)
        weights = (value @ query.transpose(1, 2)) / math.sqrt(seq.shape[-1])
        weights = torch.exp(weights)
        weights = (weights.transpose(1, 2) * mask.transpose(1, 0)).transpose(1, 2)
        weights = weights / (weights.sum(-1, keepdim=True))
        return (weights @ keys).transpose(1, 0) * mask


class TanhAttn(nn.Module):
    """
    Module that calculates self-attention weights as done in Epideep
    """

    def __init__(self, dim_in=40, value_dim=40, key_dim=40) -> None:
        """
        param dim_in: Dimensionality of input sequence
        param value_dim: Dimension of value transform
        param key_dim: Dimension of key transform
        """
        super(TanhAttn, self).__init__()
        self.value_layer = nn.Linear(dim_in, value_dim)
        self.query_layer = nn.Linear(dim_in, value_dim)
        self.key_layer = nn.Linear(dim_in, key_dim)

    def forward(self, seq):
        """
        param seq: Sequence in dimension [Seq len, Batch, Hidden size]
        """
        seq_in = seq.transpose(0, 1)
        value = self.value_layer(seq_in)
        value = torch.tanh(value)
        query = value
        keys = seq_in
        weights = value @ query.transpose(1, 2)
        weights = torch.softmax(weights, -1)
        return (weights @ keys).transpose(1, 0)

    def forward_mask(self, seq, mask):
        """
        param seq: Sequence in dimension [Seq len, Batch, Hidden size]
        """
        seq_in = seq.transpose(0, 1)
        value = self.value_layer(seq_in)
        value = torch.tanh(value)
        query = value
        keys = seq_in
        weights = value @ query.transpose(1, 2)
        weights = torch.exp(weights)
        weights = (weights.transpose(1, 2) * mask.transpose(1, 0)).transpose(1, 2)
        weights = weights / (weights.sum(-1, keepdim=True))
        return (weights @ keys).transpose(1, 0)


class LatentAtten(nn.Module):
    """
    Attention on latent representation
    """

    def __init__(self, h_dim, key_dim=None) -> None:
        super(LatentAtten, self).__init__()
        if key_dim is None:
            key_dim = h_dim
        self.key_dim = key_dim
        self.key_layer = nn.Linear(h_dim, key_dim)
        self.query_layer = nn.Linear(h_dim, key_dim)

    def forward(self, h_M, h_R):
        key = self.key_layer(h_M)
        query = self.query_layer(h_R)
        atten = (key @ query.transpose(0, 1)) / math.sqrt(self.key_dim)
        atten = torch.softmax(atten, 1)
        return atten


class EmbedAttenSeq(nn.Module):
    """
    Module to embed a sequence. Adds Attention module to
    """

    def __init__(
        self,
        dim_seq_in: int = 5,
        dim_metadata: int = 3,
        rnn_out: int = 40,
        dim_out: int = 50,
        n_layers: int = 1,
        bidirectional: bool = False,
        attn=TransformerAttn,
        dropout=0.0,
    ) -> None:
        """
        param dim_seq_in: Dimensionality of input vector (no. of age groups)
        param dim_out: Dimensionality of output vector
        param dim_metadata: Dimensions of metadata for all sequences
        param rnn_out: output dimension for rnn
        """
        super(EmbedAttenSeq, self).__init__()

        self.dim_seq_in = dim_seq_in
        self.dim_metadata = dim_metadata
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=self.rnn_out // 2 if self.bidirectional else self.rnn_out,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=dropout,
        )
        self.attn_layer = attn(self.rnn_out, self.rnn_out, self.rnn_out)
        self.out_layer = [
            nn.Linear(
                in_features=self.rnn_out + self.dim_metadata, out_features=self.dim_out
            ),
            nn.Tanh(),
            nn.Dropout(dropout),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

    def forward_mask(self, seqs, metadata, mask):
        # Take last output from GRU
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = latent_seqs
        latent_seqs = self.attn_layer.forward_mask(latent_seqs, mask)
        latent_seqs = latent_seqs.sum(0)
        out = self.out_layer(torch.cat([latent_seqs, metadata], dim=1))
        return out

    def forward(self, seqs, metadata):
        # Take last output from GRU
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = self.attn_layer(latent_seqs).sum(0)
        out = self.out_layer(torch.cat([latent_seqs, metadata], dim=1))
        return out


class EmbedAttenSeq2(nn.Module):
    """
    Module to embed a sequence. Adds Attention module to
    """

    def __init__(
        self,
        dim_seq_in: int = 5,
        dim_label_in: int = 1,
        dim_metadata: int = 3,
        rnn_out: int = 40,
        dim_out: int = 50,
        n_layers: int = 1,
        bidirectional: bool = False,
        attn=TransformerAttn,
        dropout=0.0,
    ) -> None:
        """
        param dim_seq_in: Dimensionality of input vector (no. of age groups)
        param dim_out: Dimensionality of output vector
        param dim_metadata: Dimensions of metadata for all sequences
        param rnn_out: output dimension for rnn
        """
        super(EmbedAttenSeq2, self).__init__()

        self.dim_seq_in = dim_seq_in
        self.dim_label_in = dim_label_in
        self.dim_metadata = dim_metadata
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=self.rnn_out // 2 if self.bidirectional else self.rnn_out,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=dropout,
        )
        self.attn_layer = attn(self.rnn_out, self.rnn_out, self.rnn_out)
        self.out_layer = [
            nn.Linear(
                in_features=self.rnn_out + self.dim_metadata + self.dim_label_in,
                out_features=self.dim_out,
            ),
            nn.Tanh(),
            nn.Dropout(dropout),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

    def forward_mask(self, seqs, metadata, mask, labels):
        # Take last output from GRU
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = latent_seqs
        latent_seqs = self.attn_layer.forward_mask(latent_seqs, mask)
        latent_seqs = latent_seqs.sum(0)
        out = self.out_layer(torch.cat([latent_seqs, metadata, labels], dim=1))
        return out

    def forward(self, seqs, metadata, labels):
        # Take last output from GRU
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = self.attn_layer(latent_seqs).sum(0)
        out = self.out_layer(torch.cat([latent_seqs, metadata, labels], dim=1))
        return out


class EmbedSeq(nn.Module):
    """
    Module to embed a sequence 
    """

    def __init__(
        self,
        dim_seq_in: int = 5,
        dim_metadata: int = 3,
        rnn_out: int = 40,
        dim_out: int = 50,
        n_layers: int = 1,
        bidirectional: bool = False,
    ) -> None:
        """
        param dim_seq_in: Dimensionality of input vector (no. of age groups)
        param dim_out: Dimensionality of output vector
        param dim_metadata: Dimensions of metadata for all sequences
        param rnn_out: output dimension for rnn
        """
        super(EmbedSeq, self).__init__()

        self.dim_seq_in = dim_seq_in
        self.dim_metadata = dim_metadata
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=self.rnn_out // 2 if self.bidirectional else self.rnn_out,
            bidirectional=bidirectional,
            num_layers=n_layers,
        )
        self.out_layer = [
            nn.Linear(
                in_features=self.rnn_out + self.dim_metadata, out_features=self.dim_out
            ),
            nn.Tanh(),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

    def forward_mask(self, seqs, metadata, mask):
        # Take last output from GRU
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = latent_seqs * mask
        latent_seqs = latent_seqs.sum(0)

        out = self.out_layer(torch.cat([latent_seqs, metadata], dim=1))
        return out

    def forward(self, seqs, metadata):
        # Take last output from GRU
        latent_seqs = self.rnn(seqs)[0][0]

        out = self.out_layer(torch.cat([latent_seqs, metadata], dim=1))
        return out


class EmbedSeq2(nn.Module):
    """
    Module to embed a sequence 
    """

    def __init__(
        self,
        dim_seq_in: int = 5,
        dim_label_in: int = 1,
        dim_metadata: int = 3,
        rnn_out: int = 40,
        dim_out: int = 50,
        n_layers: int = 1,
        bidirectional: bool = False,
    ) -> None:
        """
        param dim_seq_in: Dimensionality of input vector
        param dim_label_in: Dimensions of label vector (usually 1)
        param dim_out: Dimensionality of output vector
        param dim_metadata: Dimensions of metadata for all sequences
        param rnn_out: output dimension for rnn
        """
        super(EmbedSeq2, self).__init__()

        self.dim_seq_in = dim_seq_in
        self.dim_label_in = dim_label_in
        self.dim_metadata = dim_metadata
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=self.rnn_out // 2 if self.bidirectional else self.rnn_out,
            bidirectional=bidirectional,
            num_layers=n_layers,
        )
        self.out_layer = [
            nn.Linear(
                in_features=self.rnn_out + self.dim_metadata + self.dim_label_in,
                out_features=self.dim_out,
            ),
            nn.Tanh(),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

    def forward_mask(self, seqs, metadata, mask, labels):
        # Take last output from GRU
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = latent_seqs * mask
        latent_seqs = latent_seqs.sum(0)

        out = self.out_layer(torch.cat([latent_seqs, metadata, labels], dim=1))
        return out

    def forward(self, seqs, metadata, labels):
        # Take last output from GRU
        latent_seqs = self.rnn(seqs)[0][0]

        out = self.out_layer(torch.cat([latent_seqs, metadata, labels], dim=1))
        return out


class RegressionFNP(nn.Module):
    """
    Functional Neural Process for regression
    """

    def __init__(
        self,
        dim_x=1,
        dim_y=1,
        dim_h=50,
        transf_y=None,
        n_layers=1,
        use_plus=True,
        num_M=100,
        dim_u=1,
        dim_z=1,
        fb_z=0.0,
        use_ref_labels=True,
        use_DAG=True,
        add_atten=False,
    ):
        """
        :param dim_x: Dimensionality of the input
        :param dim_y: Dimensionality of the output
        :param dim_h: Dimensionality of the hidden layers
        :param transf_y: Transformation of the output (e.g. standardization)
        :param n_layers: How many hidden layers to use
        :param use_plus: Whether to use the FNP+
        :param num_M: How many points exist in the training set that are not part of the reference set
        :param dim_u: Dimensionality of the latents in the embedding space
        :param dim_z: Dimensionality of the  latents that summarize the parents
        :param fb_z: How many free bits do we allow for the latent variable z
        """
        super(RegressionFNP, self).__init__()

        self.num_M = num_M
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_h = dim_h
        self.dim_u = dim_u
        self.dim_z = dim_z
        self.use_plus = use_plus
        self.fb_z = fb_z
        self.transf_y = transf_y
        self.use_ref_labels = use_ref_labels
        self.use_DAG = use_DAG
        self.add_atten = add_atten
        # normalizes the graph such that inner products correspond to averages of the parents
        self.norm_graph = lambda x: x / (torch.sum(x, 1, keepdim=True) + 1e-8)

        self.register_buffer("lambda_z", float_tensor(1).fill_(1e-8))

        # function that assigns the edge probabilities in the graph
        self.pairwise_g_logscale = nn.Parameter(
            float_tensor(1).fill_(math.log(math.sqrt(self.dim_u)))
        )
        self.pairwise_g = lambda x: logitexp(
            -0.5
            * torch.sum(
                torch.pow(x[:, self.dim_u :] - x[:, 0 : self.dim_u], 2), 1, keepdim=True
            )
            / self.pairwise_g_logscale.exp()
        ).view(x.size(0), 1)
        # transformation of the input

        init = [nn.Linear(dim_x, self.dim_h), nn.ReLU()]
        for i in range(n_layers - 1):
            init += [nn.Linear(self.dim_h, self.dim_h), nn.ReLU()]
        self.cond_trans = nn.Sequential(*init)
        # p(u|x)
        self.p_u = nn.Linear(self.dim_h, 2 * self.dim_u)
        # q(z|x)
        self.q_z = nn.Linear(self.dim_h, 2 * self.dim_z)
        # for p(z|A, XR, yR)
        if use_ref_labels:
            self.trans_cond_y = nn.Linear(self.dim_y, 2 * self.dim_z)

        # p(y|z) or p(y|z, u)
        self.output = nn.Sequential(
            nn.Linear(
                self.dim_z if not self.use_plus else self.dim_z + self.dim_u, self.dim_h
            ),
            nn.ReLU(),
            nn.Linear(self.dim_h, 2 * dim_y),
        )
        if self.add_atten:
            self.atten_layer = LatentAtten(self.dim_h)

    def forward(self, XR, yR, XM, yM, kl_anneal=1.0):
        X_all = torch.cat([XR, XM], dim=0)
        H_all = self.cond_trans(X_all)

        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)
        u = pu.rsample()

        # get G
        if self.use_DAG:
            G = sample_DAG(u[0 : XR.size(0)], self.pairwise_g, training=self.training)
        else:
            G = sample_Clique(
                u[0 : XR.size(0)], self.pairwise_g, training=self.training
            )

        # get A
        A = sample_bipartite(
            u[XR.size(0) :], u[0 : XR.size(0)], self.pairwise_g, training=self.training
        )
        if self.add_atten:
            HR, HM = H_all[0 : XR.size(0)], H_all[XR.size(0) :]
            atten = self.atten_layer(HM, HR)
            A = A * atten

        # get Z
        qz_mean_all, qz_logscale_all = torch.split(self.q_z(H_all), self.dim_z, 1)
        qz = Normal(qz_mean_all, qz_logscale_all)
        z = qz.rsample()
        if self.use_ref_labels:
            cond_y_mean, cond_y_logscale = torch.split(
                self.trans_cond_y(yR), self.dim_z, 1
            )
            pz_mean_all = torch.mm(
                self.norm_graph(torch.cat([G, A], dim=0)),
                cond_y_mean + qz_mean_all[0 : XR.size(0)],
            )
            pz_logscale_all = torch.mm(
                self.norm_graph(torch.cat([G, A], dim=0)),
                cond_y_logscale + qz_logscale_all[0 : XR.size(0)],
            )
        else:
            pz_mean_all = torch.mm(
                self.norm_graph(torch.cat([G, A], dim=0)), qz_mean_all[0 : XR.size(0)],
            )
            pz_logscale_all = torch.mm(
                self.norm_graph(torch.cat([G, A], dim=0)),
                qz_logscale_all[0 : XR.size(0)],
            )

        pz = Normal(pz_mean_all, pz_logscale_all)

        pqz_all = pz.log_prob(z) - qz.log_prob(z)

        # apply free bits for the latent z
        if self.fb_z > 0:
            log_qpz = -torch.sum(pqz_all)

            if self.training:
                if log_qpz.item() > self.fb_z * z.size(0) * z.size(1) * (1 + 0.05):
                    self.lambda_z = torch.clamp(
                        self.lambda_z * (1 + 0.1), min=1e-8, max=1.0
                    )
                elif log_qpz.item() < self.fb_z * z.size(0) * z.size(1):
                    self.lambda_z = torch.clamp(
                        self.lambda_z * (1 - 0.1), min=1e-8, max=1.0
                    )

            log_pqz_R = self.lambda_z * torch.sum(pqz_all[0 : XR.size(0)])
            log_pqz_M = self.lambda_z * torch.sum(pqz_all[XR.size(0) :])

        else:
            log_pqz_R = torch.sum(pqz_all[0 : XR.size(0)])
            log_pqz_M = torch.sum(pqz_all[XR.size(0) :])

        final_rep = z if not self.use_plus else torch.cat([z, u], dim=1)

        mean_y, logstd_y = torch.split(self.output(final_rep), 1, dim=1)
        logstd_y = torch.log(0.1 + 0.9 * F.softplus(logstd_y))

        mean_yR, mean_yM = mean_y[0 : XR.size(0)], mean_y[XR.size(0) :]
        logstd_yR, logstd_yM = logstd_y[0 : XR.size(0)], logstd_y[XR.size(0) :]

        # logp(R)
        pyR = Normal(mean_yR, logstd_yR)
        log_pyR = torch.sum(pyR.log_prob(yR))

        # logp(M|S)
        pyM = Normal(mean_yM, logstd_yM)
        log_pyM = torch.sum(pyM.log_prob(yM))

        obj_R = (log_pyR + log_pqz_R) / float(self.num_M)
        obj_M = (log_pyM + log_pqz_M) / float(XM.size(0))

        if self.use_ref_labels:
            obj = obj_R + obj_M
        else:
            obj = obj_M

        loss = -obj

        return loss, mean_y, logstd_y

    def predict(self, x_new, XR, yR, sample=True):

        H_all = self.cond_trans(torch.cat([XR, x_new], 0))

        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)
        u = pu.rsample()

        A = sample_bipartite(
            u[XR.size(0) :], u[0 : XR.size(0)], self.pairwise_g, training=False
        )

        if self.add_atten:
            HR, HM = H_all[0 : XR.size(0)], H_all[XR.size(0) :]
            atten = self.atten_layer(HM, HR)
            A = A * atten

        pz_mean_all, pz_logscale_all = torch.split(
            self.q_z(H_all[0 : XR.size(0)]), self.dim_z, 1
        )
        if self.use_ref_labels:
            cond_y_mean, cond_y_logscale = torch.split(
                self.trans_cond_y(yR), self.dim_z, 1
            )
            pz_mean_all = torch.mm(self.norm_graph(A), cond_y_mean + pz_mean_all)
            pz_logscale_all = torch.mm(
                self.norm_graph(A), cond_y_logscale + pz_logscale_all
            )
        else:
            pz_mean_all = torch.mm(self.norm_graph(A), pz_mean_all)
            pz_logscale_all = torch.mm(self.norm_graph(A), pz_logscale_all)
        pz = Normal(pz_mean_all, pz_logscale_all)

        z = pz.rsample()
        final_rep = z if not self.use_plus else torch.cat([z, u[XR.size(0) :]], dim=1)

        mean_y, logstd_y = torch.split(self.output(final_rep), 1, dim=1)
        logstd_y = torch.log(0.1 + 0.9 * F.softplus(logstd_y))

        init_y = Normal(mean_y, logstd_y)
        if sample:
            y_new_i = init_y.sample()
        else:
            y_new_i = mean_y

        y_pred = y_new_i

        if self.transf_y is not None:
            if torch.cuda.is_available():
                y_pred = self.transf_y.inverse_transform(y_pred.cpu().data.numpy())
            else:
                y_pred = self.transf_y.inverse_transform(y_pred.data.numpy())

        return y_pred, mean_y, logstd_y, u[XR.size(0) :], u[: XR.size(0)], init_y, A


class SelfAttention(nn.Module):
    """
    Simple attention layer
    """

    def __init__(self, hidden_dim, n_heads=8):
        super(SelfAttention, self).__init__()
        self._W_k = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_heads)]
        )
        self._W_v = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_heads)]
        )
        self._W_q = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_heads)]
        )
        self._W = nn.Linear(n_heads * hidden_dim, hidden_dim)
        self.n_heads = n_heads

    def forward(self, x):
        outs = []
        for i in range(self.n_heads):
            k_ = self._W_k[i](x)
            v_ = self._W_v[i](x)
            q_ = self._W_q[i](x)
            wts = torch.softmax(v_ @ q_.T, dim=-1)
            out = wts @ k_
            outs.append(out)
        outs = torch.cat(outs, dim=-1)
        outs = self._W(outs)
        return outs


class RegressionFNP2(nn.Module):
    """
    Functional Neural Process for regression
    """

    def __init__(
        self,
        dim_x=1,
        dim_y=1,
        dim_h=50,
        transf_y=None,
        n_layers=1,
        use_plus=True,
        num_M=100,
        dim_u=1,
        dim_z=1,
        fb_z=0.0,
        use_ref_labels=True,
        use_DAG=True,
        add_atten=False,
    ):
        """
        :param dim_x: Dimensionality of the input
        :param dim_y: Dimensionality of the output
        :param dim_h: Dimensionality of the hidden layers
        :param transf_y: Transformation of the output (e.g. standardization)
        :param n_layers: How many hidden layers to use
        :param use_plus: Whether to use the FNP+
        :param num_M: How many points exist in the training set that are not part of the reference set
        :param dim_u: Dimensionality of the latents in the embedding space
        :param dim_z: Dimensionality of the  latents that summarize the parents
        :param fb_z: How many free bits do we allow for the latent variable z
        """
        super(RegressionFNP2, self).__init__()

        self.num_M = num_M
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_h = dim_h
        self.dim_u = dim_u
        self.dim_z = dim_z
        self.use_plus = use_plus
        self.fb_z = fb_z
        self.transf_y = transf_y
        self.use_ref_labels = use_ref_labels
        self.use_DAG = use_DAG
        self.add_atten = add_atten
        # normalizes the graph such that inner products correspond to averages of the parents
        self.norm_graph = lambda x: x / (torch.sum(x, 1, keepdim=True) + 1e-8)

        self.register_buffer("lambda_z", float_tensor(1).fill_(1e-8))

        # function that assigns the edge probabilities in the graph
        self.pairwise_g_logscale = nn.Parameter(
            float_tensor(1).fill_(math.log(math.sqrt(self.dim_u)))
        )
        self.pairwise_g = lambda x: logitexp(
            -0.5
            * torch.sum(
                torch.pow(x[:, self.dim_u :] - x[:, 0 : self.dim_u], 2), 1, keepdim=True
            )
            / self.pairwise_g_logscale.exp()
        ).view(x.size(0), 1)
        # transformation of the input

        init = [nn.Linear(dim_x, self.dim_h), nn.ReLU()]
        for i in range(n_layers - 1):
            init += [nn.Linear(self.dim_h, self.dim_h), nn.ReLU()]
        self.cond_trans = nn.Sequential(*init)
        # p(u|x)
        self.p_u = nn.Linear(self.dim_h, 2 * self.dim_u)
        # q(z|x)
        self.q_z = nn.Linear(self.dim_h, 2 * self.dim_z)
        # for p(z|A, XR, yR)
        if use_ref_labels:
            self.trans_cond_y = nn.Linear(self.dim_y, 2 * self.dim_z)

        # p(y|z) or p(y|z, u)
        # TODO: Add for sR input
        self.atten_ref = SelfAttention(self.dim_x)
        self.output = nn.Sequential(
            nn.Linear(
                self.dim_z + self.dim_x
                if not self.use_plus
                else self.dim_z + self.dim_u + self.dim_x,
                self.dim_h,
            ),
            nn.ReLU(),
            nn.Linear(self.dim_h, 2 * dim_y),
        )
        if self.add_atten:
            self.atten_layer = LatentAtten(self.dim_h)

    def forward(self, XR, yR, XM, yM, kl_anneal=1.0):
        # sR = self.atten_ref(XR).mean(dim=0)
        sR = XR.mean(dim=0)
        X_all = torch.cat([XR, XM], dim=0)
        H_all = self.cond_trans(X_all)

        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)
        u = pu.rsample()

        # get G
        if self.use_DAG:
            G = sample_DAG(u[0 : XR.size(0)], self.pairwise_g, training=self.training)
        else:
            G = sample_Clique(
                u[0 : XR.size(0)], self.pairwise_g, training=self.training
            )

        # get A
        A = sample_bipartite(
            u[XR.size(0) :], u[0 : XR.size(0)], self.pairwise_g, training=self.training
        )
        if self.add_atten:
            HR, HM = H_all[0 : XR.size(0)], H_all[XR.size(0) :]
            atten = self.atten_layer(HM, HR)
            A = A * atten

        # get Z
        qz_mean_all, qz_logscale_all = torch.split(self.q_z(H_all), self.dim_z, 1)
        qz = Normal(qz_mean_all, qz_logscale_all)
        z = qz.rsample()
        if self.use_ref_labels:
            cond_y_mean, cond_y_logscale = torch.split(
                self.trans_cond_y(yR), self.dim_z, 1
            )
            pz_mean_all = torch.mm(
                self.norm_graph(torch.cat([G, A], dim=0)),
                cond_y_mean + qz_mean_all[0 : XR.size(0)],
            )
            pz_logscale_all = torch.mm(
                self.norm_graph(torch.cat([G, A], dim=0)),
                cond_y_logscale + qz_logscale_all[0 : XR.size(0)],
            )
        else:
            pz_mean_all = torch.mm(
                self.norm_graph(torch.cat([G, A], dim=0)), qz_mean_all[0 : XR.size(0)],
            )
            pz_logscale_all = torch.mm(
                self.norm_graph(torch.cat([G, A], dim=0)),
                qz_logscale_all[0 : XR.size(0)],
            )

        pz = Normal(pz_mean_all, pz_logscale_all)

        pqz_all = pz.log_prob(z) - qz.log_prob(z)

        # apply free bits for the latent z
        if self.fb_z > 0:
            log_qpz = -torch.sum(pqz_all)

            if self.training:
                if log_qpz.item() > self.fb_z * z.size(0) * z.size(1) * (1 + 0.05):
                    self.lambda_z = torch.clamp(
                        self.lambda_z * (1 + 0.1), min=1e-8, max=1.0
                    )
                elif log_qpz.item() < self.fb_z * z.size(0) * z.size(1):
                    self.lambda_z = torch.clamp(
                        self.lambda_z * (1 - 0.1), min=1e-8, max=1.0
                    )

            log_pqz_R = self.lambda_z * torch.sum(pqz_all[0 : XR.size(0)])
            log_pqz_M = self.lambda_z * torch.sum(pqz_all[XR.size(0) :])

        else:
            log_pqz_R = torch.sum(pqz_all[0 : XR.size(0)])
            log_pqz_M = torch.sum(pqz_all[XR.size(0) :])

        final_rep = z if not self.use_plus else torch.cat([z, u], dim=1)
        sR = sR.repeat(final_rep.shape[0], 1)
        final_rep = torch.cat([sR, final_rep], dim=-1)

        mean_y, logstd_y = torch.split(self.output(final_rep), 1, dim=1)
        logstd_y = torch.log(0.1 + 0.9 * F.softplus(logstd_y))

        mean_yR, mean_yM = mean_y[0 : XR.size(0)], mean_y[XR.size(0) :]
        logstd_yR, logstd_yM = logstd_y[0 : XR.size(0)], logstd_y[XR.size(0) :]

        # logp(R)
        pyR = Normal(mean_yR, logstd_yR)
        log_pyR = torch.sum(pyR.log_prob(yR))

        # logp(M|S)
        pyM = Normal(mean_yM, logstd_yM)
        log_pyM = torch.sum(pyM.log_prob(yM))

        obj_R = (log_pyR + log_pqz_R) / float(self.num_M)
        obj_M = (log_pyM + log_pqz_M) / float(XM.size(0))

        if self.use_ref_labels:
            obj = obj_R + obj_M
        else:
            obj = obj_M

        loss = -obj

        return loss, mean_y, logstd_y

    def predict(self, x_new, XR, yR, sample=True):
        # sR = self.atten_ref(XR).mean(dim=0)
        sR = XR.mean(dim=0)
        H_all = self.cond_trans(torch.cat([XR, x_new], 0))

        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)
        u = pu.rsample()

        A = sample_bipartite(
            u[XR.size(0) :], u[0 : XR.size(0)], self.pairwise_g, training=False
        )

        if self.add_atten:
            HR, HM = H_all[0 : XR.size(0)], H_all[XR.size(0) :]
            atten = self.atten_layer(HM, HR)
            A = A * atten

        pz_mean_all, pz_logscale_all = torch.split(
            self.q_z(H_all[0 : XR.size(0)]), self.dim_z, 1
        )
        if self.use_ref_labels:
            cond_y_mean, cond_y_logscale = torch.split(
                self.trans_cond_y(yR), self.dim_z, 1
            )
            pz_mean_all = torch.mm(self.norm_graph(A), cond_y_mean + pz_mean_all)
            pz_logscale_all = torch.mm(
                self.norm_graph(A), cond_y_logscale + pz_logscale_all
            )
        else:
            pz_mean_all = torch.mm(self.norm_graph(A), pz_mean_all)
            pz_logscale_all = torch.mm(self.norm_graph(A), pz_logscale_all)
        pz = Normal(pz_mean_all, pz_logscale_all)

        z = pz.rsample()
        final_rep = z if not self.use_plus else torch.cat([z, u[XR.size(0) :]], dim=1)
        sR = sR.repeat(final_rep.shape[0], 1)
        final_rep = torch.cat([sR, final_rep], dim=-1)

        mean_y, logstd_y = torch.split(self.output(final_rep), 1, dim=1)
        logstd_y = torch.log(0.1 + 0.9 * F.softplus(logstd_y))

        init_y = Normal(mean_y, logstd_y)
        if sample:
            y_new_i = init_y.sample()
        else:
            y_new_i = mean_y

        y_pred = y_new_i

        if self.transf_y is not None:
            if torch.cuda.is_available():
                y_pred = self.transf_y.inverse_transform(y_pred.cpu().data.numpy())
            else:
                y_pred = self.transf_y.inverse_transform(y_pred.data.numpy())

        return y_pred, mean_y, logstd_y, u[XR.size(0) :], u[: XR.size(0)], init_y, A


class ClassificationFNP(nn.Module):
    """
    Functional Neural Process for classification with the LeNet-5 architecture
    """

    def __init__(
        self,
        dim_x=(1, 28, 28),
        dim_y=10,
        use_plus=True,
        num_M=1,
        dim_u=32,
        dim_z=64,
        fb_z=1.0,
    ):
        """
        :param dim_x: Dimensionality of the input
        :param dim_y: Dimensionality of the output
        :param use_plus: Whether to use the FNP+
        :param num_M: How many points exist in the training set that are not part of the reference set
        :param dim_u: Dimensionality of the latents in the embedding space
        :param dim_z: Dimensionality of the  latents that summarize the parents
        :param fb_z: How many free bits do we allow for the latent variable z
        """
        super(ClassificationFNP, self).__init__()

        self.num_M = num_M
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_u = dim_u
        self.dim_z = dim_z
        self.use_plus = use_plus
        self.fb_z = fb_z
        # normalizes the graph such that inner products correspond to averages of the parents
        self.norm_graph = lambda x: x / (torch.sum(x, 1, keepdim=True) + 1e-8)

        self.register_buffer("lambda_z", float_tensor(1).fill_(1e-8))

        # function that assigns the edge probabilities in the graph
        self.pairwise_g_logscale = nn.Parameter(
            float_tensor(1).fill_(math.log(math.sqrt(self.dim_u)))
        )
        self.pairwise_g = lambda x: logitexp(
            -0.5
            * torch.sum(
                torch.pow(x[:, self.dim_u :] - x[:, 0 : self.dim_u], 2), 1, keepdim=True
            )
            / self.pairwise_g_logscale.exp()
        ).view(x.size(0), 1)

        # transformation of the input
        self.cond_trans = nn.Sequential(
            nn.Conv2d(self.dim_x[0], 20, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 50, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(800, 500),
        )

        # p(u|x)
        self.p_u = nn.Sequential(nn.ReLU(), nn.Linear(500, 2 * self.dim_u))
        # q(z|x)
        self.q_z = nn.Sequential(nn.ReLU(), nn.Linear(500, 2 * self.dim_z))
        # for p(z|A, XR, yR)
        self.trans_cond_y = nn.Linear(self.dim_y, 2 * self.dim_z)

        # p(y|z) or p(y|z, u)
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                self.dim_z if not self.use_plus else self.dim_z + self.dim_u, dim_y
            ),
        )

    def forward(self, XM, yM, XR, yR, kl_anneal=1.0):
        X_all = torch.cat([XR, XM], dim=0)
        H_all = self.cond_trans(X_all)

        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)
        u = pu.rsample()

        # get G
        G = sample_DAG(u[0 : XR.size(0)], self.pairwise_g, training=self.training)

        # get A
        A = sample_bipartite(
            u[XR.size(0) :], u[0 : XR.size(0)], self.pairwise_g, training=self.training
        )

        # get Z
        qz_mean_all, qz_logscale_all = torch.split(self.q_z(H_all), self.dim_z, 1)
        qz = Normal(qz_mean_all, qz_logscale_all)
        z = qz.rsample()

        cond_y_mean, cond_y_logscale = torch.split(
            self.trans_cond_y(one_hot(yR, n_classes=self.dim_y)), self.dim_z, 1
        )
        pz_mean_all = torch.mm(
            self.norm_graph(torch.cat([G, A], dim=0)),
            cond_y_mean + qz_mean_all[0 : XR.size(0)],
        )
        pz_logscale_all = torch.mm(
            self.norm_graph(torch.cat([G, A], dim=0)),
            cond_y_logscale + qz_logscale_all[0 : XR.size(0)],
        )

        pz = Normal(pz_mean_all, pz_logscale_all)

        pqz_all = pz.log_prob(z) - qz.log_prob(z)

        # apply free bits for the latent z
        if self.fb_z > 0:
            log_qpz = -torch.sum(pqz_all)

            if self.training:
                if log_qpz.item() > self.fb_z * z.size(0) * z.size(1) * (1 + 0.05):
                    self.lambda_z = torch.clamp(
                        self.lambda_z * (1 + 0.1), min=1e-8, max=1.0
                    )
                elif log_qpz.item() < self.fb_z * z.size(0) * z.size(1):
                    self.lambda_z = torch.clamp(
                        self.lambda_z * (1 - 0.1), min=1e-8, max=1.0
                    )

            log_pqz_R = self.lambda_z * torch.sum(pqz_all[0 : XR.size(0)])
            log_pqz_M = self.lambda_z * torch.sum(pqz_all[XR.size(0) :])

        else:
            log_pqz_R = torch.sum(pqz_all[0 : XR.size(0)])
            log_pqz_M = torch.sum(pqz_all[XR.size(0) :])

        final_rep = z if not self.use_plus else torch.cat([z, u], dim=1)

        logits_all = self.output(final_rep)

        pyR = Categorical(logits=logits_all[0 : XR.size(0)])
        log_pyR = torch.sum(pyR.log_prob(yR))

        pyM = Categorical(logits=logits_all[XR.size(0) :])
        log_pyM = torch.sum(pyM.log_prob(yM))

        obj_R = (log_pyR + log_pqz_R) / float(self.num_M)
        obj_M = (log_pyM + log_pqz_M) / float(XM.size(0))

        obj = obj_R + obj_M

        loss = -obj

        return loss

    def get_pred_logits(self, x_new, XR, yR, n_samples=100):
        H_all = self.cond_trans(torch.cat([XR, x_new], 0))

        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)

        qz_mean_R, qz_logscale_R = torch.split(
            self.q_z(H_all[0 : XR.size(0)]), self.dim_z, 1
        )

        logits = float_tensor(x_new.size(0), self.dim_y, n_samples)
        for i in range(n_samples):
            u = pu.rsample()

            A = sample_bipartite(
                u[XR.size(0) :], u[0 : XR.size(0)], self.pairwise_g, training=False
            )

            cond_y_mean, cond_y_logscale = torch.split(
                self.trans_cond_y(one_hot(yR, n_classes=self.dim_y)), self.dim_z, 1
            )
            pz_mean_M = torch.mm(self.norm_graph(A), cond_y_mean + qz_mean_R)
            pz_logscale_M = torch.mm(
                self.norm_graph(A), cond_y_logscale + qz_logscale_R
            )
            pz = Normal(pz_mean_M, pz_logscale_M)

            z = pz.rsample()

            final_rep = (
                z if not self.use_plus else torch.cat([z, u[XR.size(0) :]], dim=1)
            )

            logits[:, :, i] = F.log_softmax(self.output(final_rep), 1)

        logits = torch.logsumexp(logits, 2) - math.log(n_samples)

        return logits

    def predict(self, x_new, XR, yR, n_samples=100):
        logits = self.get_pred_logits(x_new, XR, yR, n_samples=n_samples)
        return torch.argmax(logits, 1)

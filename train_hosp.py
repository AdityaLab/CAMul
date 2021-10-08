import numpy as np
import pickle
import os

from optparse import OptionParser
import networkx as nx

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from symp_extract.consts import include_cols
import dgl

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
parser.add_option("-e", "--epiweek", dest="epiweek", default="202139", type="string")
parser.add_option("-d", "--day", dest="day_ahead", default=1, type="int")
parser.add_option("-s", "--seed", dest="seed", default=0, type="int")
parser.add_option("-s", "--save", dest="save_model", default="default", type="string")

# First do sequence alone
# Then add exo features
# Then TOD (as feature, as view)
# llely demographic features

import copy
import json
import os
import sys
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import tqdm
from collections import OrderedDict
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from anndata import AnnData
import scanpy as sc
import gseapy as gp   # gene set enrichment analysis

import torch
from torchtext.vocab import Vocab
#from torchtext._torchtext import (
#    Vocab as VocabPybind,
#)

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.tasks import GeneEmbedding
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed




################################################
set_seed(42)
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
n_hvg = 1200
n_bins = 51
mask_value = -1
pad_value = -2
n_input_bins = n_bins

################################################
### Step 1: Load pre-trained model and dataset
#################################################

# Specify model path; here we load the pre-trained _scGPT blood model
model_dir = Path("../save/scGPT_bc")
model_config_file = model_dir / "args.json"
model_file = model_dir / "best_model.pt"
vocab_file = model_dir / "vocab.json"

# load gene vocabulare from saved model / inherited from torchtext.Vocabulary
vocab = GeneVocab.from_file(vocab_file)

# make sure all special tokens are included into vocabulary
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)

# Retrieve model parameters from config files
# TODO: check how to get training data from CellXgenes
with open(model_config_file, "r") as f:
    model_configs = json.load(f)

print(
    f"Resume model from {model_file}, the model args will override the "
    f"config {model_config_file}."
)

embsize = model_configs["embsize"]
nhead = model_configs["nheads"]
d_hid = model_configs["d_hid"]     # what is that?
nlayers = model_configs["nlayers"]
n_layers_cls = model_configs["n_layers_cls"]

gene2idx = vocab.get_stoi()   # get mapping tokens to indices.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = len(vocab)  # size of vocabulary

model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    vocab=vocab,
    pad_value=pad_value,
    n_input_bins=n_input_bins,
)

checkpoint = torch.load(model_file, map_location=device)    # Load OrderedDict

chk_keys = checkpoint.keys()
mdl_keys = model.state_dict().keys()
if set(chk_keys) != set(mdl_keys):
    print('----- WARNING! model and checkpoint mismatch!')
    larger_dict = chk_keys if len(chk_keys) >= len(mdl_keys) else mdl_keys
    for key in larger_dict:
        key1 = key if key in chk_keys else "N/A"
        key2 = key if key in mdl_keys else "N/A"
        print(f"{key2}  \  {key1}")  # model keys first

try:
    # @Alex: the model was saved for parallell execution, need a conversion for CPU
    model.load_state_dict(checkpoint)
    print(f"Loading all model params from {model_file}")
except Exception as e:
    print(e)

    # only load params that are in the model and match the size
    model_dict = model.state_dict()

    # dictionary containing the state of the model. This dictionary is often referred to as pretrained_dict
    # how it can be that model has mode param that dict?
    pretrained_dict = torch.load(model_file, map_location=device)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    for k, v in pretrained_dict.items():
        print(f"Loading params {k} with shape {v.shape}")

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

model.to(device)


########### 1.2 Load dataset of interest from  CellXgene
# Specify data path; here we load the Immune Human dataset
data_dir = Path("../data")

# sc is library for single-cell RNA-seq (scRNA-seq) analysis
# Clustering, DEA, Meta Analisys, Vizualization
adata = sc.read(
    str(data_dir / "Immune_ALL_human.h5ad"), cache=True
)  # 33506 × 12303

ori_batch_col = "batch"
adata.obs["celltype"] = adata.obs["final_annotation"].astype(str)
data_is_raw = False





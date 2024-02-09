import copy
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd
# from . import asyn
import pickle
import torch
from anndata import AnnData
import scanpy as sc
import seaborn as sns
import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torchtext.vocab import Vocab
from sklearn.metrics import confusion_matrix

import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics

from _conf_annot import *
from _load_data import load_annot_dataset, _harmonize_anndata_with_foundational_model
from tutorials._utils import _load_foundational_vocabulary_add_spec_tokens, get_root_folder


warnings.filterwarnings('ignore')
logger = scg.logger
if device == 'cuda':
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    torch.cuda.empty_cache()


###### Get parameters and folders paths
# We imported all preprocessing parameters from _conf_annot

params = Hyperparameters()

params.validate(
    input_style=input_style,
    output_style=output_style,
    input_emb_style=input_emb_style,
    ADV=ADV,
    DAB=DAB
)


dataset_name = Hyperparameters.dataset_name
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)

# LoaD ANNOTATION DATASET, set some variables
adata, adata_test = load_annot_dataset(dataset_name)
if dataset_name == "ms":
    data_is_raw = False
    filter_gene_by_counts = False


###### load foundational model
foundational_model_path = get_root_folder() / "save/scGPT_human"
model_config_file = foundational_model_path / "args.json"
found_model_file = foundational_model_path / "best_model.pt"
found_vocab_file = foundational_model_path / "vocab.json"

# Paths to foundational model
print(get_root_folder())
print(Hyperparameters.load_model)
model_dir = get_root_folder() / Hyperparameters.load_model
model_config_file = model_dir / "args.json"
model_file = model_dir / "best_model.pt"
vocab_file = model_dir / "vocab.json"

# Load gene vocabulary and add special tokens. Vocabulary is gene-->token mapping, "RP5-973N23.5": 60693
vocab = _load_foundational_vocabulary_add_spec_tokens(vocab_file, special_tokens)


# remove from adata those genes which are not in foundational model
adata = _harmonize_anndata_with_foundational_model(adata, vocab)


# Load foundational model parameters from json config
with open(model_config_file, "r") as f:
    model_configs = json.load(f)
logger.info(
    f"Resume model from {model_file}, the model args will override the "
    f"config {model_config_file}."
)
embsize = model_configs["embsize"]
nhead = model_configs["nheads"]
d_hid = model_configs["d_hid"]
nlayers = model_configs["nlayers"]
n_layers_cls = model_configs["n_layers_cls"]


# PRE-PROCESS data
# use the args to config the workflow: filter_gene_by_counts, data_is_raw,
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=filter_gene_by_counts,    # step 1
    filter_cell_by_counts=False,                    # step 2
    normalize_total=1e4,                            # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,                              # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=False,                               # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=n_bins,                                 # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)

adata_test = adata[adata.obs["str_batch"] == "1"]
adata = adata[adata.obs["str_batch"] == "0"]

preprocessor(adata, batch_key=None)
preprocessor(adata_test, batch_key=None)


### Train-test split in a perversive way
input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
    "normed_raw": "X_normed",
    "log1p": "X_normed",
    "binned": "X_binned",
}[input_style]

all_counts = (
    adata.layers[input_layer_key].A
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)

genes = adata.var["gene_name"].tolist()

celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
celltypes_labels = np.array(celltypes_labels)

batch_ids = adata.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
batch_ids = np.array(batch_ids)

(
    train_data,
    valid_data,
    train_celltype_labels,
    valid_celltype_labels,
    train_batch_labels,
    valid_batch_labels,
) = train_test_split(
    all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
)
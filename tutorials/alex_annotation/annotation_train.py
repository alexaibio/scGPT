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
from _load_data import load_annot_dataset

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

# LoaD ANNOTATION DATASET
ann_ds = load_annot_dataset(dataset_name)

# load foundational model



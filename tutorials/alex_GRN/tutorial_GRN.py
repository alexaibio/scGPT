import json
import os
import sys
import warnings
from pathlib import Path
import numpy as np
import seaborn as sns

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import scanpy as sc

import torch

#from torchtext._torchtext import (
#    Vocab as VocabPybind,
#)

sys.path.insert(0, "../../")
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
model_dir = Path("../../save/scGPT_bc")
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

from tutorials._utils import _compare_model_and_checkpoint
_compare_model_and_checkpoint(model, checkpoint)

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
'''
In the zero-shot setting, we extracted the gene similarity network from scGPT models’s gene embeddings based on cosine similarities. 
In the finetuned setting, we constructed the gene networks in a similar manner from the scGPT model finetuned
on the Immune Human dataset. Following Ceglia et al.
'''
# Specify data path; here we load the Immune Human dataset
data_dir = Path("data/FineTune")

# sc is library for single-cell RNA-seq (scRNA-seq) analysis: Clustering, DEA, Meta Analisys, Vizualization
# https://www.nature.com/articles/s41592-021-01336-8#data-availability
# https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_in_single-cell_genomics_-_integration_task_datasets_Immune_and_pancreas_/12420968
adata = sc.read(
    str(data_dir / "Immune_ALL_human.h5ad"), cache=True
)  # 33506 × 12303

ori_batch_col = "batch"
adata.obs["celltype"] = adata.obs["final_annotation"].astype(str)   # why?
data_is_raw = False

# Preprocess the data following the scGPT data pre-processing pipeline
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=3,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=n_hvg,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)

preprocessor(adata, batch_key="batch")



#####################################################
# Step 2: Retrieve scGPT's gene embeddings

# Overall, the pre-trained foundation model contains 30+K genes. Here for simplicity, we focus on a subset of HVGs specific to the data at hand.
# Retrieve the data-independent gene embeddings from scGPT
gene_ids = np.array([id for id in gene2idx.values()])
gene_embeddings = model.encoder(torch.tensor(gene_ids, dtype=torch.long).to(device))
gene_embeddings = gene_embeddings.detach().cpu().numpy()

# Filter on the intersection between the Immune Human HVGs found in step 1.2 and scGPT's 30+K foundation model vocab
gene_embeddings = {gene: gene_embeddings[i] for i, gene in enumerate(gene2idx.keys()) if gene in adata.var.index.tolist()}
print('Retrieved gene embeddings for {} genes.'.format(len(gene_embeddings)))

# Construct gene embedding network
embed = GeneEmbedding(gene_embeddings)

####################################################
# Step 3: Extract gene programs from gene embedding network

# Perform Louvain clustering with desired resolution; here we specify resolution=40
gdata = embed.get_adata(resolution=40)
# Retrieve the gene clusters
metagenes = embed.get_metagenes(gdata)


# Obtain the set of gene programs from clusters with #genes >= 5
mgs = dict()
for mg, genes in metagenes.items():
    if len(genes) > 4:
        mgs[mg] = genes

# Here are the gene programs identified
mgs


# visualize
sns.set(font_scale=0.35)
embed.score_metagenes(adata, metagenes)
embed.plot_metagenes_scores(adata, mgs, "celltype")



###################
# Step 5: Visualize network connectivity within desired gene program








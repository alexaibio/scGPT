import sys
import scgpt as scg
import torch
import json
from pathlib import Path
from dataclasses import dataclass, field, InitVar


logger = scg.logger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Hyperparameters:
    seed: int = 0
    dataset_name: str = "ms"
    do_train: bool = True
    load_model: str = "../save/scGPT_human"
    mask_ratio: float = 0.0
    epochs: int = 10
    n_bins: int = 51
    MVC: bool = False
    ecs_thres: float = 0.0
    dab_weight: float = 0.0
    lr: float = 1e-5
    batch_size: int = 30
    layer_size: int = 128
    nlayers: int = 4
    nhead: int = 4
    dropout: float = 0.2
    schedule_ratio: float = 0.9
    schedule_interval: int = 1
    save_eval_interval: int = 5
    fast_transformer: bool = True
    pre_norm: bool = False
    amp: bool = True
    include_zero_gene: bool = False
    freeze: bool = False
    DSBN: bool = False

    def validate(self, input_style, output_style, input_emb_style, ADV, DAB):
        assert input_style in ["normed_raw", "log1p", "binned"]
        assert output_style in ["normed_raw", "log1p", "binned"]
        assert input_emb_style in ["category", "continuous", "scaling"]

        if input_style == "binned" and input_emb_style == "scaling":
            raise ValueError("input_emb_style `scaling` is not supported for binned input.")

        if (input_style == "log1p" or input_style == "normed_raw") and input_emb_style == "category":
            raise ValueError("input_emb_style `category` is not supported for log1p or normed_raw input.")

        if input_emb_style == "category":
            mask_value = self.n_bins + 1
            pad_value = self.n_bins  # for padding gene expr values
            n_input_bins = self.n_bins + 2
        else:
            mask_value = -1
            pad_value = -2
            n_input_bins = self.n_bins

        if ADV and DAB:
            raise ValueError("ADV and DAB cannot be both True.")

        self.DAB_separate_optim = True if DAB > 1 else False



# fine tuning logging interval
log_interval = 250



###### 1 -  Training Settings

# settings for data processing

pad_token = "<pad>"
special_tokens = ["<pad>", "<cls>", "<eoc>"]  # <cls> - for aggregating all genes into a cell representation ??
mask_ratio = Hyperparameters.mask_ratio     # what is that?
mask_value = "auto"  # for masked values, now it should always be auto
include_zero_gene = Hyperparameters.include_zero_gene  # # include zero expr genes in training input, "all", "batch-wise", "row-wise",False
max_seq_len = 3001  # was 1536 for perturbation
n_bins = Hyperparameters.n_bins

# input/output representation
input_style = "binned"  # "normed_raw", "log1p", or "binned"
output_style = "binned"  # "normed_raw", "log1p", or "binned"

# settings for training
MLM = False  # whether to use masked language modeling, currently it is always on.
CLS = True         # celltype classification objective
ADV = False  # Adversarial training for batch correction
CCE = False  # Contrastive cell embedding objective
MVC = Hyperparameters.MVC  # Masked value prediction for cell embedding
ECS = Hyperparameters.ecs_thres > 0  # Elastic cell similarity objective
DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
mvc_decoder_style = "inner product"





def get_foundation_model_parameters(model_file: Path, model_config_file: Path):
    # default settings for the model
    embsize = 512   # embedding dimension
    d_hid = 512     # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 12    # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 8       # number of heads in nn.MultiheadAttention
    n_layers_cls = 3
    dropout = 0.2   # dropout probability
    use_fast_transformer = True  # whether to use fast transformer

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(f"Resume model from {model_file}, the model args will override the config {model_config_file}.")

    # substitute default parameters above with those from loaded foundational model
    logger.info(f'Default values vs loaded:')
    logger.info('         embsize  |  nhead  |  d_hid  |  nlayers  |  n_layers_cls')
    logger.info(f' before:   {embsize},       {nhead},       {d_hid},        {nlayers},     {n_layers_cls}')

    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

    logger.info(f' after:    {embsize},       {nhead},      {d_hid},        {nlayers},     {n_layers_cls}')

    return embsize, nhead, d_hid, nlayers, n_layers_cls, dropout, use_fast_transformer






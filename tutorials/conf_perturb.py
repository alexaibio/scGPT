import sys
sys.path.insert(0, "../")
import scgpt as scg
import torch
import json
logger = scg.logger


###### 1 -  Training Settings

# settings for data processing
TRN_SET = {
    'pad_token': "<pad>",
    'special_tokens': ["<pad>", "<cls>", "<eoc>"],
    'pad_value': 0,             # for padding values
    'pert_pad_id': 2,
    'n_hvg': 0,                 # number of highly variable genes
    'include_zero_gene': "all",  # include zero expr genes in training input, "all", "batch-wise", "row-wise",False
    'max_seq_len': 1536,
    # settings for training
    'MLM': True,        # whether to use masked language modeling, currently it is always on.
    'CLS': False,       # celltype classification objective
    'CCE': False,       # Contrastive cell embedding objective
    'MVC': False,       # Masked value prediction for cell embedding
    'ECS': False,  # Elastic cell similarity objective
    'cell_emb_style': "cls",
    'mvc_decoder_style': "inner product, detach",
    'amp': True
}


# settings for optimizer
OPT_SET = {
    'lr': 6e-5,             # or 1e-4
    'batch_size': 30,       # was 64
    'eval_batch_size': 30,   # was 64
    'epochs': 10,
    'schedule_interval': 1,
    'early_stop': 5
}



def get_foundation_model_parameters(model_file, model_config_file):
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
    logger.info(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )

    # substitute default config parameters with loaded ones
    logger.info(f'Default values vs loaded:')
    logger.info(' before:     embsize  |  nhead  |  d_hid  |  nlayers  |  n_layers_cls')
    logger.info(f' {embsize},   {nhead},   {d_hid},   {nlayers},   {n_layers_cls}')

    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

    logger.info(f' after:     {embsize},   {nhead},   {d_hid},   {nlayers},   {n_layers_cls}')

    return embsize, nhead, d_hid, nlayers, n_layers_cls, dropout, use_fast_transformer


# logging
log_interval = 250


data_name = "adamson"
split = "simulation"
if data_name == "norman":
    perts_to_plot = ["SAMD1+ZBTB1"]
elif data_name == "adamson":
    perts_to_plot = ["KCTD16+ctrl"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
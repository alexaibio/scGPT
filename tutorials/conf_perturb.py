import torch


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
    'lr': 5e-5,             # or 1e-4
    'batch_size': 28,       # was 64
    'eval_batch_size': 28,   # was 64
    'epochs': 10,
    'schedule_interval': 1,
    'early_stop': 5
}

# settings for the model
embsize = 512  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
n_layers_cls = 3
dropout = 0.2  # dropout probability
use_fast_transformer = True  # whether to use fast transformer

# logging
log_interval = 250


data_name = "adamson"
split = "simulation"
if data_name == "norman":
    perts_to_plot = ["SAMD1+ZBTB1"]
elif data_name == "adamson":
    perts_to_plot = ["KCTD16+ctrl"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
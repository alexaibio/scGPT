import json
import os
import sys
import time
import copy
from pathlib import Path
import warnings
import torch
import numpy as np
import matplotlib
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab
from torchtext.vocab import (Vocab as VocabPybind)

# GEARS: Predicting transcriptional outcomes of novel multi-gene perturbations
from gears import PertData

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.loss import (
    masked_mse_loss,
    criterion_neg_log_bernoulli,
    masked_relative_error,
)
from scgpt.tokenizer import tokenize_batch, pad_batch, tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id
from tutorials._train import train, evaluate
from tutorials._predict import plot_perturbation

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = Path(f"./save/dev_perturb-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"saving to {save_dir}")
logger = scg.logger
#scg.utils.add_file_handler(logger, save_dir / "run.log")


############################################################
# add if to use flash-attention
# what if fast transformer?

###### 1 -  Training Settings
if True:
    # settings for data prcocessing
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    pad_value = 0  # for padding values
    pert_pad_id = 2

    TRN_SET = {
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



    load_model = "../save/scGPT_human"
    load_param_prefixs = [
        "encoder",
        "value_encoder",
        "transformer_encoder",
    ]

    # settings for optimizer
    lr = 1e-4  # or 1e-4
    batch_size = 64
    eval_batch_size = 64
    epochs = 15
    schedule_interval = 1
    early_stop = 5

    # settings for the model
    embsize = 512  # embedding dimension
    d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 12  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 8  # number of heads in nn.MultiheadAttention
    n_layers_cls = 3
    dropout = 0.2  # dropout probability
    use_fast_transformer = True  # whether to use fast transformer

    # logging
    log_interval = 100

#############  choose a validation dataset: adamson or norman
logger.info(' Load finetuning dataset')
data_name = "adamson"
split = "simulation"
if data_name == "norman":
    perts_to_plot = ["SAMD1+ZBTB1"]
elif data_name == "adamson":
    perts_to_plot = ["KCTD16+ctrl"]


# log running date and current git commit
logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
pert_data = PertData("./data")   # downloading, from gears import PertData
pert_data.load(data_name=data_name)
pert_data.prepare_split(split=split, seed=1)
pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)


###################  Load scGPT pre-trained model
if load_model is not None:
    model_dir = Path(load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    # ???? why we need pert data here?
    pert_data.adata.var["id_in_vocab"] = [ 1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"] ]
    gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    genes = pert_data.adata.var["gene_name"].tolist()

    # load pre-trained model
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
    # ?? loaded: genes, embzize etc
else:
    genes = pert_data.adata.var["gene_name"].tolist()
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]


vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(
    [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes],
    dtype=int
)
n_genes = len(genes)

inGENE = {
    'gene_ids': gene_ids,
    'n_genes': n_genes
}



###############################
# 2 - Create and train scGpt

ntokens = len(vocab)  # size of vocabulary
model = TransformerGenerator(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=n_layers_cls,
    n_cls=1,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    pert_pad_id=pert_pad_id,
    do_mvc=TRN_SET['MVC'],
    cell_emb_style=TRN_SET['cell_emb_style'],
    mvc_decoder_style=TRN_SET['mvc_decoder_style'],
    use_fast_transformer=use_fast_transformer,
)

# can I use load_pretrained() here to avois flash-attention?
pretrained_dict = torch.load(model_file, map_location=device)

from tutorials._utils import _compare_model_and_checkpoint
_compare_model_and_checkpoint(model, pretrained_dict)

# load_param_prefixs: "encoder", "value_encoder", "transformer_encoder" - what is rthe difference?
if (load_param_prefixs is not None) and load_model is not None:
    # only load params that start with the prefix (why???? Noi decoder? no cls_decoder? no mvc_decoder)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if any([k.startswith(prefix) for prefix in load_param_prefixs])
    }
    for k, v in pretrained_dict.items():
        logger.info(f"Loading params {k} with shape {v.shape}")

    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
elif load_model is not None:  # either param_prefixed or model is None
    try:
        model.load_state_dict(torch.load(model_file))
        logger.info(f"Loading all model params from {model_file}")
    except Exception as e:
        print(e)

        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        #pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

model.to(device)

print(model)


################### FINETUNING: train and validate def here

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)
OPTM = {
    'criterion': masked_mse_loss,
    'criterion_cls': nn.CrossEntropyLoss(),
    'optimizer': optimizer,
    'scheduler': scheduler,
    'scaler': torch.cuda.amp.GradScaler(enabled=TRN_SET['amp'])
}

best_val_loss = float("inf")
best_model = None
patience = 0

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()

    # get adamson dataset for finetuning
    train_loader = pert_data.dataloader["train_loader"]
    valid_loader = pert_data.dataloader["val_loader"]

    train(
        model,
        train_loader,
        TRN_SET,
        inGENE,
        OPTM,
        log_interval,
        epoch
    )

    val_loss, val_mre = evaluate(
        model,
        valid_loader,
        TRN_SET,
        inGENE,
        OPTM
    )

    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        f"valid loss/mse {val_loss:5.4f} |"
    )
    logger.info("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        logger.info(f"Best model with score {best_val_loss:5.4f}")
        patience = 0
    else:
        patience += 1
        if patience >= early_stop:
            logger.info(f"Early stop at epoch {epoch}")
            break

    torch.save(
        model.state_dict(),
        save_dir / f"model_{epoch}.pt",
    )

    scheduler.step()



################### Predict and Plot

# predict(best_model, [["FEV"], ["FEV", "SAMD11"]])
for p in perts_to_plot:
    plot_perturbation(best_model, pert_data, p, pool_size=300, save_file=f"{save_dir}/{p}.png")


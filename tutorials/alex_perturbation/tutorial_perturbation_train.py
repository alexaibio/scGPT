import sys
import time
import copy
from pathlib import Path
import warnings
import torch
import numpy as np
import matplotlib
from torch import nn
import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.loss import masked_mse_loss
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed
from tutorials.alex_perturbation._train import train, evaluate
from tutorials.alex_perturbation._load_data import _load_perturbation_dataset, _harmonize_pert_dataset_with_foundational_model, _load_foundational_vocabulary_add_spec_tokens
from tutorials.alex_perturbation._conf_perturb import device
from _conf_perturb import (
    OPT_SET, TRN_SET,
    get_foundation_model_parameters,
    log_interval,
    data_name, split
)
from gears import PertData

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")
set_seed(42)

logger = scg.logger
#scg.utils.add_file_handler(logger, save_dir / "run.log")

# create folder for today's fine-tuning
run_save_dir = Path(f"./save/fine_tune_perturb-{time.strftime('%b%d-%H-%M')}/")
run_save_dir.mkdir(parents=True, exist_ok=True)
print(f"saving to {run_save_dir}")


if device == 'cuda':
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    torch.cuda.empty_cache()


############################################################
# TODO:
# add if to use flash-attention
# what if fast transformer?
# GEARS: https://github.com/snap-stanford/GEARS/tree/master
# gears paper: https://www.nature.com/articles/s41587-023-01905-6



######## load scGPT pre-trained model

# use pretrained model - all HUMAN or BRAIN or BLOOD
# Param prefixes are prefixes of ther layers names
foundational_model_path = "save/scGPT_human"
load_param_prefixs = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
]
model_foundational_dir = Path(foundational_model_path)
model_config_file = model_foundational_dir / "args.json"
found_model_file = model_foundational_dir / "best_model.pt"
found_vocab_file = model_foundational_dir / "vocab.json"

# model vocabulary (gene id and names):  60697 -> A1BG
vocab_foundational: GeneVocab = _load_foundational_vocabulary_add_spec_tokens(found_vocab_file)

# model config parameters...
# embsize=512, nhead=8, d_hid=512, nlayers=12, n_layers_cls=3
embsize, nhead, d_hid, nlayers, n_layers_cls, dropout, use_fast_transformer = get_foundation_model_parameters(
    found_model_file,
    model_config_file
)


###### Load and correct perturbation data
# pert_data.adata = 68603(observations) x 5060 (genes)
# why perturbations are like CREB1+ctrl - is it control should be separated?
# data_name = "adamson", split="simulation"
pert_data: PertData = _load_perturbation_dataset(data_name, split)

# in foundational model set token to <pad> if it is not in perturbation gene tokens
gene_ids: np.ndarray
n_genes_pert: int
gene_ids, n_genes_pert, pert_data = _harmonize_pert_dataset_with_foundational_model(pert_data, vocab_foundational)

# create data structure to pass as a parameter
inGENE = {
    'gene_ids': gene_ids,   # np.ndarray
    'n_genes': n_genes_pert # int
}



###############################
# 2 - Instantiate and load pre-trained foundational models and then do fine-tuning

ntokens = len(vocab_foundational)  # size of vocabulary
model = TransformerGenerator(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=n_layers_cls,
    n_cls=1,
    vocab=vocab_foundational,
    dropout=dropout,
    pad_token=TRN_SET['pad_token'],
    pad_value=TRN_SET['pad_value'],
    pert_pad_id=TRN_SET['pert_pad_id'],
    do_mvc=TRN_SET['MVC'],
    cell_emb_style=TRN_SET['cell_emb_style'],
    mvc_decoder_style=TRN_SET['mvc_decoder_style'],
    use_fast_transformer=use_fast_transformer,
)

# TODO: could use load_pretrained() here to avoid flash-attention
pretrained_dict = torch.load(found_model_file, map_location=device)

# uncomment for model debug to compare the layers
#from tutorials._utils import _compare_model_and_checkpoint
#_compare_model_and_checkpoint(model, pretrained_dict)

# filer by load_param_prefixes: remove layers which are not in param prefixes (WHY?)
# load_param_prefixs: "encoder", "value_encoder", "transformer_encoder" - what is rthe difference?

# load that dictionary into a model
if (load_param_prefixs is not None) and foundational_model_path is not None:
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
elif foundational_model_path is not None:  # either param_prefixed or model is None
    try:
        model.load_state_dict(torch.load(found_model_file))
        logger.info(f"Loading all model params from {found_model_file}")
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

optimizer = torch.optim.Adam(model.parameters(), lr=OPT_SET['lr'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, OPT_SET['schedule_interval'], gamma=0.9)
OPTM_PARAM = {
    'criterion': masked_mse_loss,         # NOTE: must be changed for every particular fine-tuning task
    'criterion_cls': nn.CrossEntropyLoss(),
    'optimizer': optimizer,
    'scheduler': scheduler,
    'scaler': torch.cuda.amp.GradScaler(enabled=TRN_SET['amp'])
}

best_val_loss = float("inf")
best_model = None
patience = 0

for current_epoch in range(1, OPT_SET['epochs'] + 1):
    epoch_start_time = time.time()

    # get adamson dataset for fine-tuning
    train_loader = pert_data.dataloader["train_loader"]
    valid_loader = pert_data.dataloader["val_loader"]

    train(
        model,
        train_loader,
        TRN_SET,
        inGENE,
        OPTM_PARAM,
        log_interval,
        current_epoch
    )

    val_loss, val_mre = evaluate(
        model,
        valid_loader,
        TRN_SET,
        inGENE,
        OPTM_PARAM
    )

    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {current_epoch:3d} | time: {elapsed:5.2f}s | "
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
        if patience >= OPT_SET['early_stop']:
            logger.info(f"Early stop at epoch {current_epoch}")
            break

    run_save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        model.state_dict(),
        run_save_dir / f"model_epoch_{current_epoch}_val_loss_{val_loss:5.4f}.pt",
    )

    scheduler.step()

logger.info("  ****** Fine-tuning of the foundational model has been completed! *****")





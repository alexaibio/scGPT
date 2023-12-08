import json
import os
import sys
import time
import copy
from pathlib import Path
import warnings
import torch
import matplotlib
from torch import nn
from torchtext.vocab import Vocab
from torchtext.vocab import (Vocab as VocabPybind)

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
from tutorials._load_data import _load_perturbation_dataset, _harmonize_pert_dataset
from tutorials.conf_perturb import device
from conf_perturb import (
    OPT_SET, TRN_SET,
    get_foundation_model_parameters,
    log_interval,
    data_name, split, perts_to_plot
)
from tutorials._load_data import _load_vocabulary_from_foundational

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")
set_seed(42)

# initialize logger
logger = scg.logger
#scg.utils.add_file_handler(logger, save_dir / "run.log")

# create folder for today's finetuning
save_dir = Path(f"./save/fine_tune_perturb-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"saving to {save_dir}")

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

# pretrained model
folder_foundational_model = "../save/scGPT_human"
load_param_prefixs = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
]

model_foundational_dir = Path(folder_foundational_model)
model_config_file = model_foundational_dir / "args.json"
model_file = model_foundational_dir / "best_model.pt"
vocab_file = model_foundational_dir / "vocab.json"

# model vocabulary...
vocab_foundational = _load_vocabulary_from_foundational(folder_foundational_model)    # 60697, gene names: A1BG etc

# model config parameters...
embsize, nhead, d_hid, nlayers, n_layers_cls, dropout, use_fast_transformer = get_foundation_model_parameters(
    model_file,
    model_config_file
)


###### Load and correct perturbation data
# original data
pert_data = _load_perturbation_dataset(data_name, split)
gene_ids, n_genes_pert, pert_data = _harmonize_pert_dataset(pert_data, vocab_foundational)

inGENE = {
    'gene_ids': gene_ids,
    'n_genes': n_genes_pert
}



###############################
# 2 - Create and train scGpt

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

# can I use load_pretrained() here to avois flash-attention?
pretrained_dict = torch.load(model_file, map_location=device)

#from tutorials._utils import _compare_model_and_checkpoint
#_compare_model_and_checkpoint(model, pretrained_dict)

# load_param_prefixs: "encoder", "value_encoder", "transformer_encoder" - what is rthe difference?
if (load_param_prefixs is not None) and folder_foundational_model is not None:
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
elif folder_foundational_model is not None:  # either param_prefixed or model is None
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

optimizer = torch.optim.Adam(model.parameters(), lr=OPT_SET['lr'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, OPT_SET['schedule_interval'], gamma=0.9)
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

for epoch in range(1, OPT_SET['epochs'] + 1):
    epoch_start_time = time.time()

    # get adamson dataset for fine-tuning
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
        if patience >= OPT_SET['early_stop']:
            logger.info(f"Early stop at epoch {epoch}")
            break

    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        model.state_dict(),
        save_dir / f"model_epoch_{epoch}_val_loss_{val_loss:5.4f}.pt",
    )

    scheduler.step()






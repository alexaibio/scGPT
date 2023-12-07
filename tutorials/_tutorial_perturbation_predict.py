from pathlib import Path
import torch
from scgpt.model import TransformerGenerator
from tutorials._predict import plot_perturbation, predict
from tutorials.conf_perturb import perts_to_plot
from tutorials._load_data import _load_vocabulary_foundational
from conf_perturb import (
    OPT_SET, TRN_SET,
    embsize, d_hid, nlayers, nhead, n_layers_cls, dropout, use_fast_transformer,
    log_interval,
    data_name, split, perts_to_plot
)
from tutorials.conf_perturb import device

################### Predict and Plot
# Load model
vocab_foundational = _load_vocabulary_foundational()

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

# load model dictionary
save_dir = Path(f"./save/fine_tune_perturb-Dec06-09-31/")
tuned_model_file = save_dir / 'model_10.pt'
best_tuned_model_dict = torch.load(tuned_model_file, map_location=device)

best_tuned_model =

# predict
predict(
    model=best_tuned_model,
    pert_list=[["FEV"], ["FEV", "SAMD11"]]
)

# plot
pert_data = None


for p in perts_to_plot:
    plot_perturbation(best_tuned_model, pert_data, p, pool_size=300, save_plot_file=f"{save_dir}/{p}.png")
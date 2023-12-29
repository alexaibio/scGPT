import numpy as np
import torch
import scgpt as scg
from scgpt.model import TransformerGenerator
from tutorials.alex_perturbation._predict import plot_perturbation, predict
from tutorials.alex_perturbation._load_data import (
    _load_foundational_vocabulary_add_spec_tokens,
    _harmonize_pert_dataset_with_foundational_model,
    _load_perturbation_dataset
)
from tutorials.alex_perturbation._conf_perturb import (
    INPT_PAR,
    get_foundation_model_parameters,
    perts_to_plot
)
from _conf_perturb import (
    TRN_PAR, INPT_PAR, perturbation_data_source, split
)
from tutorials.alex_perturbation._conf_perturb import device
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from _utils import get_perturb_folder, get_root_folder
logger = scg.logger


# load vocabulary, model parameters and model itself
foundational_model_path = get_root_folder() / "save/scGPT_human"
model_config_file = foundational_model_path / "args.json"
found_model_file = foundational_model_path / "best_model.pt"
found_vocab_file = foundational_model_path / "vocab.json"

# model vocabulary:  60697, gene names: A1BG etc
vocab_foundational: GeneVocab = _load_foundational_vocabulary_add_spec_tokens(found_vocab_file)

# model config parameters...
embsize, nhead, d_hid, nlayers, n_layers_cls, dropout, use_fast_transformer = get_foundation_model_parameters(
    found_model_file,
    model_config_file
)

# load what????
pert_data = _load_perturbation_dataset(perturbation_data_source, split)
gene_ids: np.ndarray
n_genes_pert: int
gene_ids, n_genes_pert, pert_data = _harmonize_pert_dataset_with_foundational_model(pert_data, vocab_foundational)

model = TransformerGenerator(
    ntoken=len(vocab_foundational),  # size of vocabulary,
    d_model= embsize,
    nhead=nhead,
    d_hid=d_hid,
    nlayers=nlayers,
    nlayers_cls=n_layers_cls,
    n_cls=1,
    vocab=vocab_foundational,
    dropout=dropout,
    pad_token=INPT_PAR['pad_token'],
    pad_value=INPT_PAR['pad_value'],
    pert_pad_id=INPT_PAR['pert_pad_id'],
    do_mvc=INPT_PAR['MVC'],
    cell_emb_style=INPT_PAR['cell_emb_style'],
    mvc_decoder_style=INPT_PAR['mvc_decoder_style'],
    use_fast_transformer=use_fast_transformer
)

############### load finetuned model - not fundamental!
run_name = "fine_tune_perturb-Dec11-20-03"
best_model = "model_epoch_10_val_loss_0.1331.pt"

INPT_PAR['run_name'] = run_name
run_save_dir = get_perturb_folder() / "save" / run_name
tuned_model_file = run_save_dir / best_model

# update weights of the model
# TODO: do we need that? compare the model before and after... it seems like it is the same
model_dict = model.state_dict()
best_tuned_model_dict = torch.load(tuned_model_file, map_location=device)
model_dict.update(best_tuned_model_dict)
model.load_state_dict(model_dict)
model.to(device)

###### test prediction of expression after perturbation
perturbed_genes_list = [["FEV"], ["FEV", "SAMD11"]]
logger.info(f'------->  Predict a perturbation for :  {perturbed_genes_list}')
predicted_expression_dict = predict(
    model=model,
    gene_ids=gene_ids,
    pert_list=perturbed_genes_list,
    pool_size=700   # remove to see all
)
# dict of FEB: ndarray[5060,], / FEV_SAMD11: (5060,)


############################ plot
logger.info(f' -----> Plot a perturbation for :  {perts_to_plot}')

for pert in perts_to_plot:
    plot_perturbation(
        model=model,
        vocab_foundational=vocab_foundational,
        query=pert,
        pool_size=700,  # number of single sell to calculate / show
        save_plot_file=f"{run_save_dir}/{pert}.png"
    )

logger.info(' END of prediction and plotting')


from pathlib import Path
import torch
from scgpt.model import TransformerGenerator
from tutorials._predict import plot_perturbation, predict
from tutorials.conf_perturb import perts_to_plot
from tutorials._load_data import _load_vocabulary_from_foundational
from conf_perturb import (
    OPT_SET, TRN_SET,
    get_foundation_model_parameters,
    log_interval,
    data_name, split, perts_to_plot
)
from tutorials.conf_perturb import device
from scgpt.tokenizer.gene_tokenizer import GeneVocab


################### Predict and Plot

# load vocabulary, model parameters and model itself
foundational_model_path = "../save/scGPT_human"
model_foundational_dir = Path(foundational_model_path)
model_config_file = model_foundational_dir / "args.json"
found_model_file = model_foundational_dir / "best_model.pt"
found_vocab_file = model_foundational_dir / "vocab.json"

# model vocabulary:  60697, gene names: A1BG etc
vocab_foundational: GeneVocab = _load_vocabulary_from_foundational(found_vocab_file)

# model config parameters...
embsize, nhead, d_hid, nlayers, n_layers_cls, dropout, use_fast_transformer = get_foundation_model_parameters(
    found_model_file,
    model_config_file
)


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

############### load fine tuned model - not fundamental!
run_save_dir = Path(f"./save/fine_tune_perturb-Dec06-09-31/")
tuned_model_file = run_save_dir / 'model_10.pt'
best_tuned_model_dict = torch.load(tuned_model_file, map_location=device)

model_dict = model.state_dict()
model_dict.update(best_tuned_model_dict)
model.load_state_dict(model_dict)
model.to(device)



# predict expression after perturbation
results_pred = predict(
    model=model,
    vocab_foundational=vocab_foundational,
    pert_list=[["FEV"], ["FEV", "SAMD11"]]
)


# plot
for pert in perts_to_plot:
    plot_perturbation(
        model=model,
        vocab_foundational=vocab_foundational,
        query=pert,
        pool_size=300,
        save_plot_file=f"{run_save_dir}/{pert}.png"
    )
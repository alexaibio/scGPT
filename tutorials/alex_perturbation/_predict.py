import numpy as np
import torch
from torch import nn
from typing import List, Dict, Optional
from torch_geometric.loader import DataLoader
from gears.utils import create_cell_graph_dataset_for_prediction
import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from tutorials.alex_perturbation._load_data import _load_perturbation_dataset, _harmonize_pert_dataset_with_foundational_model
from _conf_perturb import (
    TRN_PAR, INPT_PAR
)
from gears import PertData
import seaborn as sns
import matplotlib.pyplot as plt

logger = scg.logger
sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)


def predict(
    model: TransformerGenerator,
    gene_ids,
    pert_list: List[str],
    pool_size: Optional[int] = None
) -> Dict:
    """
    Predict the gene expression values for the given perturbations.

    Args:
        model (:class:`torch.nn.Module`): The model to use for prediction.
        pert_list (:obj:`List[str]`): The list of perturbations to predict.
        pool_size (:obj:`int`, optional): For each perturbation, use this number
            of cells in the control and predict their perturbation results. Report
            the stats of these predictions. If `None`, use all control cells.
    """


    adata = pert_data.adata
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]

    # For each perturbation, use this number of cells in the control and predict their perturbation results.
    if pool_size is None:
        pool_size = len(ctrl_adata.obs)

    gene_list = pert_data.gene_names.values.tolist()

    # check if genes to be perturbed are in model's gene list
    if any(i not in gene_list for pert in pert_list for i in pert):
        raise ValueError("One or more genes are not in the perturbation graph. Please select from GEARS.gene_list!")

    model.eval()
    device = next(model.parameters()).device

    # run prediction
    with torch.no_grad():
        results_pred = {}
        for pert in pert_list:
            logger.info(f'... running prediction for genes {pert}')
            # GEARs (Gene Expression Analysis with Recurrent neural networkS)
            cell_graphs = create_cell_graph_dataset_for_prediction(
                pert_gene=pert,
                ctrl_adata=ctrl_adata,
                gene_names=gene_list,
                device=device,
                num_samples=pool_size
            )

            loader = DataLoader(dataset=cell_graphs, batch_size=TRN_PAR['eval_batch_size'], shuffle=False)

            preds = []
            for batch_data in loader:
                pred_gene_values = model.pred_perturb(
                    batch_data,
                    INPT_PAR['include_zero_gene'],
                    gene_ids=gene_ids,
                    amp=INPT_PAR['amp']
                )
                preds.append(pred_gene_values)
            preds = torch.cat(preds, dim=0)
            results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)

    return results_pred



def plot_perturbation(
        model: nn.Module,
        vocab_foundational: GeneVocab,
        query: str,
        save_plot_file: str = None,
        pool_size: int = None,
) -> None:

    pert_data: PertData = _load_perturbation_dataset(perturbation_data_source, split)
    gene_ids: np.ndarray
    n_genes_pert: int
    gene_ids, n_genes_pert, pert_data = _harmonize_pert_dataset_with_foundational_model(pert_data, vocab_foundational)

    adata = pert_data.adata
    gene2idx = pert_data.node_map
    cond2name = dict(adata.obs[["condition", "condition_name"]].values)
    gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

    de_idx = [
        gene2idx[gene_raw2id[i]]
        for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
    ]

    genes = [
        gene_raw2id[i] for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
    ]

    truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]

    # do PREDICTION
    if query.split("+")[1] == "ctrl":
        logger.info('... Control is present')
        pred = predict(
            model=model,
            vocab_foundational=vocab_foundational,
            pert_list=[[query.split("+")[0]]],
            pool_size=pool_size
        )
        pred = pred[query.split("+")[0]][de_idx]
    else:
        logger.info('... No control')
        pred = predict(
            model=model,
            vocab_foundational=vocab_foundational,
            pert_list=[query.split("+")],
            pool_size=pool_size
        )
        pred = pred["_".join(query.split("+"))][de_idx]

    ctrl_means = adata[adata.obs["condition"] == "ctrl"].to_df().mean()[de_idx].values
    pred = pred - ctrl_means
    truth = truth - ctrl_means

    plt.figure(figsize=[16.5, 4.5])
    plt.title(query)
    plt.boxplot(truth, showfliers=False, medianprops=dict(linewidth=0))

    for i in range(pred.shape[0]):
        _ = plt.scatter(i + 1, pred[i], color="red")

    plt.axhline(0, linestyle="dashed", color="green")

    ax = plt.gca()
    ax.xaxis.set_ticklabels(genes, rotation=90)

    plt.ylabel("Change in Gene Expression over Control", labelpad=10)
    plt.tick_params(axis="x", which="major", pad=5)
    plt.tick_params(axis="y", which="major", pad=5)
    sns.despine()

    if save_plot_file:
        plt.savefig(save_plot_file, bbox_inches="tight", transparent=False)
    # plt.show()
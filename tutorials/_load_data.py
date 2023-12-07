import time
import numpy as np
import scgpt as scg
from conf_perturb import (
    OPT_SET, TRN_SET,
    embsize, d_hid, nlayers, nhead, n_layers_cls, dropout, use_fast_transformer,
    log_interval
)
# GEARS: Predicting transcriptional outcomes of novel multi-gene perturbations
from gears import PertData
logger = scg.logger


def _load_perturbation_dataset(data_name, split):
    # log running date and current git commit
    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
    pert_data = PertData("./data")  # downloading, from gears import PertData

    # 65 899 (observations) x 5060 (genes)
    # each observation has attributes: condition (ctrl, CREB1+ctrl, ZNF326+ctrl... ), cell type (K562), observation name.   pert_data.adata.obs
    # each gene (with gene names)
    pert_data.load(
        data_name=data_name)  # 65 899 (obs_names=AAACATACACCGAT-1, cell barcode) x 5060 (var_names=ENSG00000228463 ...)
    pert_data.prepare_split(split=split, seed=1)
    pert_data.get_dataloader(batch_size=OPT_SET['batch_size'], test_batch_size=OPT_SET['eval_batch_size'])

    return pert_data


def _harmonize_pert_dataset(pert_data, vocab_foundational):
    # add an "id_in_vocab" as 1 if it is in the gene list of pre-trained foundational model and -1 otherwise
    pert_data.adata.var["id_in_vocab"] = [1 if gene in vocab_foundational else -1 for gene in
                                          pert_data.adata.var["gene_name"]]

    # print how much genes in pert dataset in original foundational model
    gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab_foundational)}."
    )

    genes_pert_dataset = pert_data.adata.var["gene_name"].tolist()

    vocab_foundational.set_default_index(vocab_foundational["<pad>"])
    # if gene in pert-dataset is not in foundational vocabulary, substitute it with <pad> token
    gene_ids = np.array(
        [vocab_foundational[gene] if gene in vocab_foundational else vocab_foundational["<pad>"] for gene in
         genes_pert_dataset],
        dtype=int
    )
    n_genes_pert = len(genes_pert_dataset)

    return gene_ids, n_genes_pert, pert_data



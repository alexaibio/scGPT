import time
import numpy as np
from pathlib import Path
import scgpt as scg
from _conf_perturb import TRN_PAR, INPT_PAR
from scgpt.tokenizer.gene_tokenizer import GeneVocab
# GEARS: Predicting transcriptional outcomes of novel multi-gene perturbations
from gears import PertData
logger = scg.logger


def _load_perturbation_dataset(data_name: str, split: str) -> PertData:
    # log running date and current git commit
    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
    pert_data = PertData("./data")  # downloading, from gears import PertData

    # observations 65 899 (obs_names=AAACATACACCGAT-1, cell barcode) x tokens 5060 (var_names=ENSG00000228463 ...)
    # each observation has attributes:
    #   condition (ctrl, CREB1+ctrl, ZNF326+ctrl... ), cell type (K562), observation name.   pert_data.adata.obs
    pert_data.load(data_name=data_name)
    pert_data.prepare_split(split=split, seed=1)
    pert_data.get_dataloader(batch_size=TRN_PAR['batch_size'], test_batch_size=TRN_PAR['eval_batch_size'])

    return pert_data


def _harmonize_pert_dataset_with_foundational_model(pert_data: PertData, vocab_foundational: GeneVocab):
    # add an "id_in_vocab" as 1 if it is in the gene list of pre-trained foundational model and -1 otherwise
    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab_foundational else -1 for gene in pert_data.adata.var["gene_name"]
    ]

    # print how much genes in pert dataset is in original foundational model also
    gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab_foundational)}."
    )

    genes_pert_dataset = pert_data.adata.var["gene_name"].tolist()

    # if gene in pert-dataset is not in foundational vocabulary, substitute it with <pad> token
    vocab_foundational.set_default_index(vocab_foundational["<pad>"])
    gene_ids = np.array(
        [vocab_foundational[gene] if gene in vocab_foundational else vocab_foundational["<pad>"] for gene in
         genes_pert_dataset],
        dtype=int
    )
    n_genes_pert = len(genes_pert_dataset)

    return gene_ids, n_genes_pert, pert_data


def _load_foundational_vocabulary_add_spec_tokens(vocab_file: Path) -> GeneVocab:

    # load vocabulary from saved file.
    vocab_foundational = GeneVocab.from_file(vocab_file)  # 60697, gene names: A1BG etc

    # add special tokes if they are still not there
    for s in INPT_PAR['special_tokens']:
        if s not in vocab_foundational:
            vocab_foundational.append_token(s)

    return vocab_foundational



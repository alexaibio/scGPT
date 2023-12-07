import time
import scgpt as scg
from conf_perturb import (
    OPT_SET, TRN_SET,
    embsize, d_hid, nlayers, nhead, n_layers_cls, dropout, use_fast_transformer,
    log_interval
)
# GEARS: Predicting transcriptional outcomes of novel multi-gene perturbations
from gears import PertData
logger = scg.logger


def load_perturbation_dataset(data_name, split):
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






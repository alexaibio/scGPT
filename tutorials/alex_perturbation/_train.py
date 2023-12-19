import sys
import time
import numpy as np
import warnings
import torch
from torch import nn
from scgpt.loss import (
    masked_mse_loss,
    criterion_neg_log_bernoulli,
    masked_relative_error,
)
from scgpt.utils import set_seed, map_raw_id_to_vocab_id
import scgpt as scg

logger = scg.logger
sys.path.insert(0, "../")
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader, # torch_geometric.DataLoader
        TRN_SET: dict,  # pad token ..., MLM / CLS ..., cell_emb_style
        inGENE: dict,   # input gene vector: gene_id, ngenes
        OPTM: dict,     # objective function: 'criterion': masked_mse_loss, etc
        log_interval: int,   # 250
        epoch: 1           # current epoch
) -> None:
    """
    Train the model for one epoch.
    """

    scaler, optimizer, criterion, scheduler = OPTM['scaler'], OPTM['optimizer'], OPTM['criterion'], OPTM['scheduler']

    model.train()   # Dropout behave differently during train and evaluation
    total_loss, total_mse = 0.0, 0.0
    start_time = time.time()

    # uncomment for  SANITY: check if / next(iter(train_loader)) / has a dimension of 2? gears must be 0.0.3 version

    num_batches = len(train_loader)

    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)

        #  (batch_size * n_genes, 2) / 30x5060=151,800 x 2 (expr, pert_flag)
        x: torch.Tensor = batch_data.x

        # solving issue with x[,1] - https://github.com/bowang-lab/scGPT/issues/101
        # old pert data gave expression x[:, 0] and perturbation flag x[:, 1]
        # this manipulation is because 'gear' package changed format of data
        # i.e. in current (new) version both perturbation and expression is included in one line, now we sepaarate them
        ori_gene_values = x[:, 0].view(batch_size, inGENE['n_genes'])  # reshape it back to [30, 5060]
        pert_flags = x[:, 1].long().view(batch_size, inGENE['n_genes'])

        target_gene_values = batch_data.y  # (batch_size, n_genes), 30x5060

        if TRN_SET['include_zero_gene'] in ["all", "batch-wise"]:  # what is include zero gene?
            if TRN_SET['include_zero_gene'] == "all":
                input_gene_ids = torch.arange(inGENE['n_genes'], device=device, dtype=torch.long)
            else:
                input_gene_ids = ( ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0] )

            # what is seq lenth in this context? it is not a language!
            # sample input_gene_id
            if len(input_gene_ids) > TRN_SET['max_seq_len']:  # len(input_gene_ids): 5060 / 'max_seq_len': 1536
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[ :TRN_SET['max_seq_len']]

            input_values = ori_gene_values[:, input_gene_ids]  # now it is only 1536 - why do we cut it?!
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, inGENE['gene_ids'])
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )
        # Automatic Mixed Precision (AMP): faster training times and reduced memory usage, especially with NVIDIA GPU
        with torch.cuda.amp.autocast(enabled=TRN_SET['amp']):
            # one forward step?
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=TRN_SET['CLS'],  # by setting one of those we select an objective function, all False
                CCE=TRN_SET['CCE'],  # except default MLM=True
                MVC=TRN_SET['MVC'],
                ECS=TRN_SET['ECS'],
            )
            output_values = output_dict["mlm_output"]   # get default MLM output

            masked_positions = torch.ones_like( input_values, dtype=torch.bool)  # Use all
            loss = loss_mse = criterion(output_values, target_values, masked_positions)

        model.zero_grad()   #  zeroing out the gradients before starting a new backward pass
        scaler.scale(loss).backward()   # Scaling the loss is a common practice in mixed-precision training.
        scaler.unscale_(optimizer)

        # dealing with warning during backpropagation
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        # uncomment?
        torch.cuda.empty_cache()

        total_loss += loss.item()
        total_mse += loss_mse.item()

        # show progress bar
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:06.5f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:6.3f} | mse {cur_mse:6.3f} |"
            )
            total_loss = 0
            total_mse = 0
            start_time = time.time()


def evaluate(model: nn.Module, val_loader: torch.utils.data.DataLoader, TRN_SET, inGENE, OPTM) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    criterion = OPTM['criterion']

    model.eval()
    total_loss = 0.0
    total_error = 0.0

    with torch.no_grad():
        for batch, batch_data in enumerate(val_loader):
            batch_size = len(batch_data.y)
            batch_data.to(device)
            x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
            ori_gene_values = x[:, 0].view(batch_size, inGENE['n_genes'])
            pert_flags = x[:, 1].long().view(batch_size, inGENE['n_genes'])
            target_gene_values = batch_data.y  # (batch_size, n_genes)

            if TRN_SET['include_zero_gene'] in ["all", "batch-wise"]:
                if TRN_SET['include_zero_gene'] == "all":
                    input_gene_ids = torch.arange(inGENE['n_genes'], device=device)
                else:  # when batch-wise
                    input_gene_ids = (
                        ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                    )

                # sample input_gene_id
                if len(input_gene_ids) > TRN_SET['max_seq_len']:
                    input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[ :TRN_SET['max_seq_len']]
                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]
                target_values = target_gene_values[:, input_gene_ids]

                mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, inGENE['gene_ids'])
                mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
                src_key_padding_mask = torch.zeros_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )
            with torch.cuda.amp.autocast(enabled=TRN_SET['amp']):
                output_dict = model(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=TRN_SET['CLS'],
                    CCE=TRN_SET['CCE'],
                    MVC=TRN_SET['MVC'],
                    ECS=TRN_SET['ECS'],
                    do_sample=True,
                )
                output_values = output_dict["mlm_output"]

                masked_positions = torch.ones_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )
                loss = criterion(output_values, target_values, masked_positions)
            total_loss += loss.item()
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item()

    return total_loss / len(val_loader), total_error / len(val_loader)


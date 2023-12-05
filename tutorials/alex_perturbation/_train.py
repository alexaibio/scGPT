import sys
import time
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


def train(model: nn.Module, train_loader: torch.utils.data.DataLoader, TRN_SET: dict, inGENE, OPTM, log_interval, epoch) -> None:
    """
    Train the model for one epoch.
    """
    scaler, optimizer, criterion, scheduler = OPTM['scaler'], OPTM['optimizer'], OPTM['criterion'], OPTM['scheduler']

    model.train()
    total_loss, total_mse = 0.0, 0.0
    start_time = time.time()

    num_batches = len(train_loader)
    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)

        ori_gene_values = x[:, 0].view(batch_size, inGENE['n_genes'])
        pert_flags = x[:, 1].long().view(batch_size, inGENE['n_genes'])
        target_gene_values = batch_data.y  # (batch_size, n_genes)

        if TRN_SET['include_zero_gene'] in ["all", "batch-wise"]:
            if TRN_SET['include_zero_gene'] == "all":
                input_gene_ids = torch.arange(inGENE['n_genes'], device=device, dtype=torch.long)
            else:
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
                input_values, dtype=torch.bool, device=device
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
            )
            output_values = output_dict["mlm_output"]

            masked_positions = torch.ones_like(
                input_values, dtype=torch.bool
            )  # Use all
            loss = loss_mse = criterion(output_values, target_values, masked_positions)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
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
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
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


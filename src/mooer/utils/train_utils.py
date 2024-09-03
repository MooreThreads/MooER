import time
from contextlib import nullcontext
import torch.distributed as dist

try:
    import torch_musa
except ImportError as e:
    print("You should install torch_musa if you want to run on Moore Threads GPU")
from mooer.utils.utils import *
from mooer.utils.checpoint_io import save_model_checkpoint_deepspeed

logger = logging.getLogger(__name__)

device = str(get_device())


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        logger.info(f"Clearing GPU cache for all ranks")
    if 'musa' in device:
        torch.musa.empty_cache()
    else:
        torch.cuda.empty_cache()
        

def train(
    model,
    train_dataloader,
    train_config,
    local_rank=None,
    rank=None,
    train_data_set=None,
    model_org=None
):
    epoch_times = []
    total_step = 0
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_context = model.join
    else:
        model_context = nullcontext
    for epoch in range(train_config.resume_epoch, train_config.num_epochs):
        if train_data_set is not None:
            dist.barrier()
            logging.info(f"RANK:{rank} Reset Dataset Epoch {epoch}...")
            train_data_set.set_epoch(epoch)
            train_dataloader_iterator = iter(train_dataloader)
            dist.barrier()
        epoch_start_time = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        step = 0
        input_dtype = torch.float32
        if train_config.use_fp16:
            input_dtype = torch.float16
        elif train_config.use_bf16:
            input_dtype = torch.bfloat16
        logging.info(f"Input data type: {input_dtype}")
        with model_context():
            should_continue = True
            while should_continue:
                try:
                    batch = next(train_dataloader_iterator)
                    total_step += 1
                    for key in batch.keys():
                        batch[key] = (
                            batch[key].to(local_rank).to(input_dtype)
                            if isinstance(batch[key], torch.Tensor)
                               and batch[key].dtype == torch.float32
                            else (
                                batch[key].to(local_rank)
                                if isinstance(batch[key], torch.Tensor)
                                else batch[key]
                            )
                        )
                    outputs, acc = model(**batch)
                    loss = outputs.loss
                    
                    acc_report = acc
                    loss_report = loss.detach().float()
                    
                    total_loss += loss.detach().float()
                    total_acc += acc
                    
                    model.backward(loss)
                    model.step()
                    
                    current_lr = model.optimizer.param_groups[0]['lr']
                    if rank == 0 and step % train_config.log_interval == 0:
                        logging.info(
                            f"Training Epoch: {epoch + 1}/{train_config.num_epochs}, "
                            f"step {step} lr {current_lr} "
                            f"completed (loss: {loss_report}, "
                            f"acc: {acc_report})")
                    
                    if step % train_config.save_interval == 0:
                        checkpoint_name = f"{train_config.model_name}_epoch_{str(epoch + 1)}_step_{step + 1}"
                        save_model_checkpoint_deepspeed(
                            model, train_config, checkpoint_name,
                            merge_rank=train_config.get('save_merge_rank', True),
                            model=model_org
                        )
                    step += 1
                
                except Exception as e:
                    logging.error(f"Exception occurred on Rank {rank}: {e}")
                    epoch_end_time = time.perf_counter() - epoch_start_time
                    logging.info(f"Epoch {epoch + 1}, Cost Time: {epoch_end_time}")
                    epoch_times.append(epoch_end_time)
                    # save model
                    checkpoint_name = f"{train_config.model_name}_epoch_{str(epoch + 2)}_step_1"
                    save_model_checkpoint_deepspeed(
                        model, train_config, checkpoint_name,
                        merge_rank=train_config.get('save_merge_rank', True),
                        model=model_org
                    )
                    break

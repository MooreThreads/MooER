import os
import random
import logging
import argparse

import deepspeed

# nn
import torch
try:
    import torch_musa
except ImportError as e:
    print("You should install torch_musa if you want to run on Moore Threads GPU")

from mooer.models import mooer_model
from mooer.utils.utils import get_device
from mooer.utils.config_utils import parse_asr_configs
from mooer.utils.train_utils import train, clear_gpu_cache
from mooer.datasets.speech_dataset_shard import SpeechDatasetShard


def parse_args():
    parser = argparse.ArgumentParser(description='DeepSpeed Training Script')
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--training_config', type=str, required=True, help='Path to the training configuration file.')
    args = parser.parse_args()
    return args


def main():
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode='w'
    )
    
    args = parse_args()
    device = str(get_device())
    
    configs = parse_asr_configs(args.training_config)
    train_config = configs['TrainConfig']
    model_config = configs['ModelConfig']
    dataset_config = configs['DataConfig']
    peft_config = configs['PeftConfig']
    deepspeed_config = train_config.deepspeed_config
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    logger.info(f"local_rank: {local_rank}, rank: {rank}, world_size: {world_size}")
    
    if 'musa' in device:
        torch.musa.manual_seed(train_config.seed + train_config.resume_epoch)
        torch.manual_seed(train_config.seed + train_config.resume_epoch)
        random.seed(train_config.seed + train_config.resume_epoch)
        torch.musa.set_device(local_rank)
    else:
        torch.cuda.manual_seed(train_config.seed + train_config.resume_epoch)
        torch.manual_seed(train_config.seed + train_config.resume_epoch)
        random.seed(train_config.seed + train_config.resume_epoch)
        torch.cuda.set_device(local_rank)
    clear_gpu_cache(local_rank)
    
    model, tokenizer = mooer_model.init_model(
        model_config=model_config,
        train_config=train_config, peft_config=peft_config)
    
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=parameters,
        config=deepspeed_config,
    )
    
    if dataset_config.get('train_data_type', 'shard') == 'shard':
        logging.info("Use shard for training...")
        dataset_train_items = SpeechDatasetShard(dataset_config=dataset_config,
                                                 tokenizer=tokenizer,
                                                 normalize=dataset_config.normalize,
                                                 mel_size=dataset_config.mel_size)
        dataset_train = dataset_train_items.dataset(
            data_type='shard',
            data_list_file=dataset_config['train_data_path'],
            shuffle=True,
            partition=True
        )

        train_dl_kwargs = {
            'batch_size': train_config.batch_size_training,
            'drop_last': True,
            'collate_fn': dataset_train_items.collator
        }
    
        train_dataloader = torch.utils.data.DataLoader(
            dataset_train,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **train_dl_kwargs,
        )
    else:
        raise KeyError
    
    # Start the training process
    train(
        model_engine,
        train_dataloader,
        train_config,
        local_rank=local_rank,
        rank=rank,
        train_data_set=dataset_train if dataset_config.get('train_data_type', 'shard') == 'shard' else None,
        model_org=model
    )


if __name__ == "__main__":
    main()
    

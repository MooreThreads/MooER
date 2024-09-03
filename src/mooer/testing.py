import os
import time
import logging
import argparse

# nn
import torch

try:
    import torch_musa
except ImportError as e:
    print("You should install torch_musa if you want to run on Moore Threads GPU")

from mooer.models import mooer_model
from mooer.utils.utils import get_device
from mooer.utils.config_utils import parse_asr_configs
from mooer.datasets.speech_dataset_shard import SpeechDatasetShard


def parse_args():
    parser = argparse.ArgumentParser(description='DeepSpeed Training Script')
    parser.add_argument('--test_config', type=str, required=True, help='Path to the testing configuration file.')
    parser.add_argument('--test_data_dir', type=str, default='', help='Path to the testing sets.')
    parser.add_argument('--test_sets', type=str, default='', help='test_sets in test_data_dir, e.g, aishell1/aishell2')
    parser.add_argument('--decode_path', type=str, required=True, help='Path to save decode text and compute wer')
    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode='w'
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    args = parse_args()
    device = str(get_device())
    
    configs = parse_asr_configs(args.test_config)
    train_config = configs['TrainConfig']
    model_config = configs['ModelConfig']
    dataset_config = configs['DataConfig']
    # reset test epoch
    logger.info("set epoch_num=1 for testing")
    dataset_config.num_epochs = 1
    
    # update paths
    if os.path.exists(args.test_data_dir):
        dataset_config.test_data_dir = args.test_data_dir
        dataset_config.test_sets = args.test_sets
    
    os.makedirs(args.decode_path, exist_ok=True)
    
    
    model, tokenizer = mooer_model.init_model(
        model_config=model_config,
        train_config=train_config)
    
    model.to(device)
    model.eval()
    
    # dataset_config = generate_dataset_config(train_config, kwargs)
    logger.info("dataset_config: {}".format(dataset_config))
    
    test_data_dir = dataset_config.test_data_dir
    test_sets = dataset_config.test_sets
    decode_path = args.decode_path
    
    for test_set in test_sets.strip().split('/'):
        test_set_path = os.path.join(test_data_dir, test_set, "data.list")
        decode_dir = os.path.join(decode_path, test_set)
        os.makedirs(decode_dir, exist_ok=True)
        logging.info(f"Test for {test_set_path}")
        if dataset_config.get('test_data_type', 'shard') == 'shard':
            logging.info("Use shard for training...")
            dataset_test_items = SpeechDatasetShard(dataset_config=dataset_config,
                                                    tokenizer=tokenizer,
                                                    normalize=dataset_config.normalize,
                                                    mel_size=dataset_config.mel_size)
            dataset_test = dataset_test_items.dataset(
                data_type='shard',
                data_list_file=test_set_path,
                shuffle=False,
                partition=False
            )
            collator = dataset_test_items.collator
        
            test_dataloader = torch.utils.data.DataLoader(
                dataset_test,
                num_workers=train_config.num_workers_dataloader,
                pin_memory=True,
                shuffle=False,
                batch_size=train_config.val_batch_size,
                drop_last=False,
                collate_fn=collator
            )
        
        else:
            raise KeyError
        
        logger.info("=====================================")
        pred_path = os.path.join(decode_dir, 'text')
        ss = time.perf_counter()
        dtype = torch.float32
        if train_config.use_fp16:
            dtype = torch.float16
        elif train_config.use_bf16:
            dtype = torch.bfloat16
        logging.info(f"Input data type: {dtype}")
        with torch.no_grad():
            with open(pred_path, "w") as pred:
                for step, batch in enumerate(test_dataloader):
                    for key in batch.keys():
                        batch[key] = batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]
                    with torch.cuda.amp.autocast(dtype=dtype):
                        model_outputs = model.generate(**batch)
                    output_text = model.tokenizer.batch_decode(model_outputs, add_special_tokens=False,
                                                               skip_special_tokens=True)
                    for key, text, target in zip(batch["keys"], output_text, batch["targets"]):
                        logging.info(f"{key} {text}")
                        pred.write(key + "\t" + text + "\n")
        logging.info(f"Infer {test_set} Cost: {time.perf_counter() - ss}")


if __name__ == "__main__":
    main()

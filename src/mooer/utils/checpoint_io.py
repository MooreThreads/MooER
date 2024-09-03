import os
import logging
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

logger = logging.getLogger(__name__)


def save_model_checkpoint_deepspeed(model_engine, cfg, checkpoint_name="checkpoint", merge_rank=False, model=None):
    logger.info(f"--> saving model ...")
    save_dir = os.path.join(cfg.output_dir, checkpoint_name)
    os.makedirs(save_dir, exist_ok=True)
    save_full_path = save_dir
    model_engine.save_checkpoint(save_dir=save_full_path, exclude_frozen_parameters=True)
    logger.info(f"encoder saved at {save_full_path}")
    
    # if merged, it will be fast when decoding
    if merge_rank:
        assert model is not None
        save_dir_merge = os.path.join(cfg.output_dir, checkpoint_name + '_merged')
        os.makedirs(save_dir_merge, exist_ok=True)
        logger.info("CKPT: loading DeepSpeed Model from: {}".format(save_full_path))
        ckpt_dict = get_fp32_state_dict_from_zero_checkpoint(save_full_path)
        logging.info("Merge Zero3 model to FP32...")
        logging.info("Save Lora Weights...")
        model.llm.save_pretrained(os.path.join(save_dir_merge, 'new_llm'))
        logging.info(f"Save finished... {os.path.join(save_dir_merge, 'new_llm')}")
        ckpt_dict_new = {}
        for key in ckpt_dict.keys():
            if 'llm' not in key:
                ckpt_dict_new[key] = ckpt_dict[key].to('cpu').clone()
            torch.save(ckpt_dict_new, os.path.join(save_dir_merge, 'adapter_project.pt'))
        logging.info(f"Save finished... {os.path.join(save_dir_merge, 'adapter_project.pt')}")
    

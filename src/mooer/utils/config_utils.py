import sys
import os
import logging
from importlib import import_module

from peft import (
    LoraConfig,
    AdaptionPromptConfig,
    PrefixTuningConfig,
)


def parse_asr_configs(file_path):
    file_dir = os.path.dirname(file_path)
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    sys.path.insert(0, file_dir)
    module = import_module(module_name)
    sys.path.pop(0)
    
    ModelConfig = getattr(module, 'ModelConfig', None)
    PeftConfig = getattr(module, 'PeftConfig', None)
    TrainConfig = getattr(module, 'TrainConfig', None)
    DataConfig = getattr(module, 'DataConfig', None)
    update_function = getattr(module, 'update', None)
    
    if None in (ModelConfig, PeftConfig, TrainConfig, DataConfig, update_function):
        raise ImportError(f"Could not find all expected classes or function in {file_path}")
    
    model_config_instance = ModelConfig()
    peft_config_instance = PeftConfig()
    train_config_instance = TrainConfig()
    data_config_instance = DataConfig()
    
    # update something
    update_function(model_config_instance,
                    train_config_instance,
                    data_config_instance)

    items = {
        'ModelConfig': model_config_instance,
        'PeftConfig': peft_config_instance,
        'TrainConfig': train_config_instance,
        'DataConfig': data_config_instance,
        'update_function': update_function
    }
    
    for key in items.keys():
        logging.info(f"################# {key} #################")
        instance = items[key]
        if isinstance(instance, (ModelConfig, PeftConfig, TrainConfig, DataConfig)):
            for attr_name, attr_value in vars(instance).items():
                logging.info(f"{attr_name}: {attr_value}")
    
    return items


def generate_peft_config(peft_config):
    peft_configs = {"lora": LoraConfig,
                    "llama_adapter": AdaptionPromptConfig,
                    "prefix": PrefixTuningConfig
                    }
    params = {}
    for attr_name, attr_value in vars(peft_config).items():
        params[attr_name] = attr_value
    params.pop("peft_method", None)
    peft_config_parse = peft_configs[peft_config.get("peft_method", "lora")](**params)

    return peft_config_parse

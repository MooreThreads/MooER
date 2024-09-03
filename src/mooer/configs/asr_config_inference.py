from dataclasses import dataclass
from typing import Optional, List
    

@dataclass
class ModelConfig:
    def __init__(self):
        self.llm_name: str = "qwen2_7b_chat"
        # You should set your own path
        self.llm_path: str = "pretrained_models/Qwen2-7B-Instruct"
        self.encoder_path: str = "pretrained_models/paraformer_encoder/paraformer-encoder.pth"
        self.adapter_path: str = "pretrained_models/asr_ast_mtl/adapter_project.pt"
        self.lora_dir: str = "pretrained_models/asr_ast_mtl/lora_weights"
        self.cmvn_path: str = "/root/MooER/src/mooer/configs/am.mvn"
        self.prompt_key: str = 'asr'  # asr, ast... you can add tasks in src/mooer/utils/data_utils.py
        ###############################
        self.llm_type: str = "decoder_only"
        self.llm_dim: int = 3584
        self.load_dtype: str = "bfloat16"
        self.encoder_name: str = 'paraformer'
        self.encoder_dim: int = 512
        self.adapter: str = "linear"
        self.adapter_downsample_rate: int = 2
        self.modal: str = "audio"
        self.normalize: Optional[bool] = False
        self.gradient_checkpoint: bool = False
        self.is_inference: bool = True
        self.prompt_template_key: str = 'qwen'

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, attribute_name, default_value=None):
        return getattr(self, attribute_name, default_value)


@dataclass
class PeftConfig:
    def __init__(self):
        self.peft_method: str = "lora"  # None , llama_adapter, prefix
        self.r: int = 64
        self.lora_alpha: int = 16
        self.target_modules: List = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "gate_proj",
                "down_proj",
            ]
        self.bias: str = "none"
        self.task_type: str = "CAUSAL_LM"
        self.lora_dropout: float = 0.05
        self.inference_mode: bool = False
        
    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, attribute_name, default_value=None):
        return getattr(self, attribute_name, default_value)


@dataclass
class TrainConfig:
    def __init__(self):
        self.model_name: str = "asr"
        self.enable_deepspeed: bool = True
        self.batch_size_training: int = 8  # you should set same as deepspeed config for throughput
        self.batching_strategy: str = 'custom'
        self.context_length: int = 4096
        self.num_epochs: int = 10
        self.num_workers_dataloader: int = 4
        
        # please set it in deepspeed config
        # self.warmup_steps: int = 1000
        # self.total_steps: int = 1000000
        # self.lr: float = 1e-4
        # self.weight_decay: float = 0.0
    
        self.save_interval: int = 20000
        self.save_merge_rank: bool = True
        # will merge deepspeed model from several rank
        self.log_interval: int = 100
        self.resume_step: int = 0
        self.resume_epoch: int = 0
        self.gamma: float = 0.85
        self.seed: int = 42
        self.use_fp16: bool = False
        self.use_bf16: bool = True
        self.mixed_precision: bool = True
        self.val_batch_size: int = 10
        self.use_peft: bool = True
        self.output_dir: str = "output/save_models"
        self.freeze_llm: bool = True
        self.freeze_encoder: bool = True
        self.freeze_projector: bool = True
        self.find_unused_parameters: bool = False
        self.gradient_checkpoint: bool = False
        self.deepspeed_config: str = '/root/MooER/src/mooer/configs/deepspeed_config_zero2.json'
        # if you want large bsz or to reduce memory, use zero3, but it will be slow
        
    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, attribute_name, default_value=None):
        return getattr(self, attribute_name, default_value)


@dataclass
class DataConfig:
    def __init__(self):
        self.train_data_path: Optional[str] = ''
        self.val_data_path: Optional[str] = ''
        self.test_data_dir: str = '/Your/testsets/root'
        self.test_sets: str = 'test-clean/test-other/aishell'
        # you can put a series of test sets under test_data_dir for testing, use / for split
        self.decode_path: Optional[str] = ''
        self.fix_length_audio: int = -1
        self.max_length: int = 2000
        self.min_length: int = 20
        self.mel_size: int = 80
        self.train_data_type: str = 'shard'
        self.test_data_type: str = 'shard'
        self.prompt_template_key: str = 'qwen'
        self.prompt_key: str = 'asr'
        self.w2v_bert_path: str = ''
        self.sort: bool = False
        self.replace_text_path: str = ''
        self.replace_type: str = 'replace'
        # you can use replace_text_path & replace_type to train other task, e.g, AST, with same uttid but different label
        
    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, attribute_name, default_value=None):
        return getattr(self, attribute_name, default_value)


def update(model_config, train_config, data_config):
    train_config.is_inference = model_config.is_inference
    data_config.is_inference = model_config.is_inference
    data_config.num_epochs = train_config.num_epochs
    data_config.adapter_downsample_rate = model_config.adapter_downsample_rate
    data_config.cmvn_path = model_config.cmvn_path
    data_config.encoder_name = model_config.encoder_name
    data_config.normalize = model_config.normalize

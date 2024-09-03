from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    def __init__(self):
        self.llm_name: str = "qwen2_7b_chat"
        # You should set your own path
        self.llm_path: str = "pretrained_models/Qwen2-7B-Instruct"
        self.encoder_path: str = "pretrained_models/paraformer_encoder/paraformer-encoder.pth"
        self.adapter_path: str = "pretrained_models/asr_ast_mtl/adapter_project.pt"
        self.lora_dir: str = "pretrained_models/asr_ast_mtl/lora_weights"
        self.cmvn_path: str = "pretrained_models/paraformer_encoder/am.mvn"
        self.prompt_key: str = 'ast'  # or asr for ASR model
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

import os
import torch
import torch.nn as nn
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
from peft import PeftModel
from mooer.utils.utils import print_module_size
from mooer.models.encoder import WhisperWrappedEncoder, HubertEncoder, W2vBert2Encoder, ParaformerEncoder
from mooer.models.adapter import LinearAdapter


import logging
logger = logging.getLogger(__name__)


def init_model(model_config):
    tokenizer = setup_tokenizer(model_config)
    encoder = setup_encoder(model_config)
    adapter = setup_adapter(model_config)
    llm = setup_llm(model_config)
    
    model = MooerModel(
        encoder,
        llm,
        adapter,
        tokenizer,
        model_config
    )
    
    # load adapter
    ckpt_path = model_config.get("adapter_path", "")
    if os.path.isdir(ckpt_path):
        logger.info("CKPT: loading DeepSpeed Model from: {}".format(ckpt_path))
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        ckpt_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)
        logging.info("Merge Zero3 model to FP32...")
        #### check params if need
        model_state_dict = model.state_dict()
        missing_keys = [k for k in ckpt_dict.keys() if k not in model_state_dict]
        for key in missing_keys:
            logging.info(f"MISSING KEY: {key}")
        model.load_state_dict(ckpt_dict, strict=False)
        if model_config.get('save_lora_weights', False):
            logging.info("Save Lora Weights...")
            model.llm.save_pretrained(os.path.join(ckpt_path, 'new_llm'))
            logging.info("Save finished...")
            exit()
    elif os.path.exists(ckpt_path):
        logger.info("CKPT: loading other parts from: {}".format(ckpt_path))
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        model_state_dict = model.state_dict()
        missing_keys = [k for k in ckpt_dict.keys() if k not in model_state_dict]
        for key in missing_keys:
            logging.info(f"MISSING KEY: {key}")
        model.load_state_dict(ckpt_dict, strict=False)
    return model, tokenizer
    

def setup_tokenizer(model_config):
    if "qwen" in model_config.llm_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.llm_path,
            padding_side="right",
            use_fast=False,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_config.llm_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def setup_encoder(model_config):
    encoder_name = model_config.encoder_name
    if encoder_name == "whisper":
        encoder = WhisperWrappedEncoder.load(model_config)
    elif encoder_name == "hubert":
        encoder = HubertEncoder.load(model_config)
    elif encoder_name == "w2v_bert2.0":
        encoder = W2vBert2Encoder.load(model_config)
    elif encoder_name == "paraformer":
        encoder = ParaformerEncoder.load(model_config)
    else:
        raise KeyError(f"not support encoder: {encoder_name}")
    print_module_size(encoder, encoder_name, 0, "====Total Params====")
    for name, param in encoder.named_parameters():
        param.requires_grad = False
    encoder.eval()
    print_module_size(encoder, encoder_name, 0, "====Trainable Params====")
    return encoder


def setup_llm(model_config):
    if model_config.load_dtype == "float16":
        load_dtype = torch.float16
    elif model_config.load_dtype == "bfloat16":
        load_dtype = torch.bfloat16
    else:
        load_dtype = torch.float32
    if "qwen" in model_config.llm_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_config.llm_path,
            use_cache=None,
            torch_dtype=load_dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.llm_path,
            use_cache=None,
            torch_dtype=load_dtype,
        )
    print_module_size(model, model_config.llm_name, 0, "====Total Params====")
    for name, param in model.named_parameters():
        param.requires_grad = False
    model.eval()
    print_module_size(model, model_config.llm_name, 0, "====Trainable Params====")
    if model_config.get("lora_dir", None) and os.path.exists(model_config.get("lora_dir", "")):
        if model_config.is_inference:
            logger.info("Inference load lora...")
            logger.info("loading lora_dir from: {}".format(model_config.get("lora_dir")))
            model = PeftModel.from_pretrained(model=model, model_id=model_config.get("lora_dir"), is_trainable=False)
            logger.info("Start Merging LLM and Adaptor...")
            model = model.merge_and_unload()
            model.eval()
            logger.info("Finish Merging LLM and Adaptor...")
        else:
            logger.info("Training load lora...")
            logger.info("loading lora_dir from: {}".format(model_config.get("lora_dir")))
            model = PeftModel.from_pretrained(model=model, model_id=model_config.get("lora_dir"), is_trainable=True)
            logger.info("Start Merging LLM and Adaptor...")
            model = model.merge_and_unload()
            logger.info("Finish Merging LLM and Adaptor...")
    return model


def setup_adapter(model_config):
    if model_config.adapter == "linear":
        adapter = LinearAdapter(model_config)
    else:
        raise KeyError(f"not support {model_config.adapter}")
    print_module_size(adapter, model_config.adapter, 0, "====Total Params====")
    for name, param in adapter.named_parameters():
        param.requires_grad = False
    adapter.eval()
    print_module_size(adapter, model_config.adapter, 0, "====Trainable Params====")
    return adapter


class MooerModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        llm: nn.Module,
        adapter: nn.Module,
        tokenizer,
        model_config,
    ):
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.encoder_projector = adapter
        self.tokenizer = tokenizer
        self.model_config = model_config
    
    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                **kwargs,
                ):
        audio_mel = kwargs.get("audio_mel", None)
        audio_mel_post_mask = kwargs.get("audio_mel_post_mask", None)
        
        audio = kwargs.get("audio", None)
        audio_mask = kwargs.get("audio_mask", None)
        visual = kwargs.get("visual", None)
        
        modality_mask = kwargs.get("modality_mask", None)
        
        # for paraformer
        audio_mel_reallen = kwargs.get("audio_mel_reallen", None)

        gradient_checkpoint = self.model_config.get("gradient_checkpoint", False)
        
        encoder_outs = None
        if audio_mel is not None or audio is not None or visual is not None:
            self.encoder.eval()
            if self.model_config.encoder_name == "whisper":
                encoder_outs = self.encoder.extract_variable_length_features(
                    audio_mel.permute(0, 2, 1), gradient_checkpoint=gradient_checkpoint)  # bs*seq*dim
            if self.model_config.encoder_name == "hubert":
                results = self.encoder(source=audio, padding_mask=1 - audio_mask)
                if self.model_config.encoder_type == "pretrain":
                    encoder_outs, audio_mel_post_mask = results["x"], results["padding_mask"]
                if self.model_config.encoder_type == "finetune":
                    encoder_outs, audio_mel_post_mask = results["encoder_out"], results["padding_mask"]
                    encoder_outs = encoder_outs.transpose(0, 1)
            if self.model_config.encoder_name == 'w2v_bert2.0':
                encoder_outs = self.encoder.extract_features(source=audio_mel, attention_mask=audio_mel_post_mask)
            if self.model_config.encoder_name == 'paraformer':
                encoder_outs = self.encoder.extract_features(source=audio_mel, reallen=audio_mel_reallen)
            if self.encoder is None:
                encoder_outs = audio_mel if audio_mel is not None else audio
            
            if self.model_config.adapter == "linear":
                encoder_outs = self.encoder_projector(encoder_outs, gradient_checkpoint=gradient_checkpoint)
        
        if input_ids is not None:
            input_ids[input_ids == -1] = 0
            if isinstance(self.llm, T5ForConditionalGeneration):
                inputs_embeds = self.llm.shared(input_ids)
            else:
                if hasattr(self.llm.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.embed_tokens(input_ids)
                elif hasattr(self.llm.model.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.model.embed_tokens(input_ids)
                else:
                    inputs_embeds = self.llm.model.model.model.embed_tokens(input_ids)
        
        if modality_mask is not None:
            modality_mask_start_indices = (modality_mask == True).float().argmax(dim=1)
            modality_lengths = torch.clamp(modality_mask.sum(dim=1), max=encoder_outs.shape[1]).tolist()
            
            encoder_outs_pad = torch.zeros_like(inputs_embeds)
            for i in range(encoder_outs.shape[0]):
                encoder_outs_pad[
                i, modality_mask_start_indices[i]:modality_mask_start_indices[i] + modality_lengths[i]
                ] = encoder_outs[i][:modality_lengths[i]]
            
            inputs_embeds = encoder_outs_pad + inputs_embeds * (~modality_mask[:, :, None])
        
        if self.model_config.get("is_inference", False):
            return inputs_embeds, attention_mask

    @torch.no_grad()
    def generate(self,
                 input_ids: torch.LongTensor = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 position_ids: Optional[torch.LongTensor] = None,
                 past_key_values: Optional[List[torch.FloatTensor]] = None,
                 inputs_embeds: Optional[torch.FloatTensor] = None,
                 labels: Optional[torch.LongTensor] = None,
                 use_cache: Optional[bool] = None,
                 output_attentions: Optional[bool] = None,
                 output_hidden_states: Optional[bool] = None,
                 return_dict: Optional[bool] = None,
                 compute_llm=True,
                 **kwargs,
                 ):
        kwargs["inference_mode"] = True
        
        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        
        if not compute_llm:
            return inputs_embeds, attention_mask, kwargs
        
        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=kwargs.get("max_new_tokens", 200),
            num_beams=kwargs.get("num_beams", 4),
            do_sample=kwargs.get("do_sample", False),
            min_length=kwargs.get("min_length", 1),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 0.8),
            temperature=kwargs.get("temperature", 1.0),
            attention_mask=attention_mask,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            sequence_bias={tuple([self.tokenizer.eos_token_id]):-0.2}
        )
        return model_outputs

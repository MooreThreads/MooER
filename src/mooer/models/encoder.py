import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class WhisperWrappedEncoder:
    
    @classmethod
    def load(cls, model_config):
        
        def extract_variable_length_features(self, x: torch.Tensor, gradient_checkpoint=False):
            """
            x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
                the mel spectrogram of the audio
            """
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
            x = x.permute(0, 2, 1)

            # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
            # x = (x + self.positional_embedding).to(x.dtype)
            x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

            for block in self.blocks:
                if gradient_checkpoint:
                    x = checkpoint(block, x)
                else:
                    x = block(x)
            if gradient_checkpoint:
                x = checkpoint(self.ln_post, x)
            else:
                x = self.ln_post(x)
            return x

        import whisper
        encoder = whisper.load_model(name=model_config.encoder_path, device='cpu').encoder
        encoder.extract_variable_length_features = types.MethodType(extract_variable_length_features, encoder)
        return encoder


class HubertEncoder:
    @classmethod
    def load(cls, model_config):
        import fairseq
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        model = models[0]
        if model_config.encoder_type == "pretrain":
            pass
        elif model_config.encoder_type == "finetune":
            model.w2v_encoder.proj = None
            model.w2v_encoder.apply_mask = False
        else:
            assert model_config.encoder_type in ["pretrain", "finetune"], "input_type must be one of [pretrain, finetune]" 
        return model
    

class W2vBert2Encoder(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        
    @classmethod
    def load(cls, model_config):
        from transformers import Wav2Vec2BertModel
        model = Wav2Vec2BertModel.from_pretrained(model_config.encoder_path)
        return cls(model_config, model)
    
    def extract_features(self, source, attention_mask):
        output = self.model(source, attention_mask=attention_mask)
        return output.last_hidden_state


class HfTextEncoder:
    @classmethod
    def load(cls, model_config):
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_config.encoder_path)
        return model
    

class ParaformerEncoder(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
    
    @classmethod
    def load(cls, model_config):
        from .Paraformer.encoder import SANMEncoder
        model = SANMEncoder(gradient_checkpoint=model_config.get('gradient_checkpoint', False))
        ckpt_dict = torch.load(model_config.encoder_path, map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)
        return cls(model_config, model)
    
    def extract_features(self, source, reallen):
        output, _, _ = self.model(
            xs_pad=source,
            ilens=reallen
        )
        # TODO: support streaming @zhenlin.liang
        return output

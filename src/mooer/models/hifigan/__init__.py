import json
import os

import torch
import torchaudio

from .hifigan import Generator
from .speaker_encoder import ResNetSpeakerEncoder


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_hifigan_model(vocoder_path, device,
                      vocoder_config=None, decoder_dim=1024):
    if vocoder_config is None:
        vocoder_config = os.path.join(os.path.dirname(__file__), 'hifigan_config.json')
    print(f"Loading vocoder config from {vocoder_config}")
    with open(vocoder_config, "r") as f:
        config = json.load(f)
    config = AttrDict(config)
    config.input_num_mels = decoder_dim
    vocoder = Generator(config)
    print(f"Loading vocoder from {vocoder_path}")
    ckpt = torch.load(vocoder_path, map_location=device)
    vocoder.load_state_dict(ckpt["generator"], strict=True)
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)
    return vocoder


def get_speaker_encoder(model_checkpoint, device):
    model = ResNetSpeakerEncoder(
        input_dim=64,
        proj_dim=512,
        log_input=True,
        use_torch_spec=True,
        audio_config={
            "fft_size": 512,
            "win_length": 400,
            "hop_length": 160,
            "sample_rate": 16000,
            "preemphasis": 0.97,
            "num_mels": 64
        },
    )
    print(f"Loading speaker encoder from {model_checkpoint}")
    checkpoint = torch.load(model_checkpoint, map_location=device)["speaker_encoder"]
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    model.to(device)
    return model

def encode_prompt_wav(spk_encoder, prompt_wav, device):
    wav, sr = torchaudio.load(prompt_wav)
    wav_16k = torchaudio.functional.resample(wav, sr, 16000)
    vocoder_latent = spk_encoder.forward(
        wav_16k.to(device), l2_norm=True
    ).unsqueeze(-1)
    return vocoder_latent


def save_wav(hifigan_generator, embedding, latent, save_path):
    samples = hifigan_generator(latent.permute(0, 2, 1), g=embedding).squeeze(0)
    samples = samples.detach().cpu()
    torchaudio.save(save_path, samples, 22050)

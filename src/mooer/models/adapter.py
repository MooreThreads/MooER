import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class LinearAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.adapter_downsample_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, config.llm_dim)

    def forward(self, x, gradient_checkpoint=False):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        
        if gradient_checkpoint:
            x = checkpoint(self.linear1, x)
        else:
            x = self.linear1(x)
        x = self.relu(x)
        if gradient_checkpoint:
            x = checkpoint(self.linear2, x)
        else:
            x = self.linear2(x)
        return x

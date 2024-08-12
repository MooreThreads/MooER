import torch
import numpy as np
import torchaudio.compliance.kaldi as kaldi


def compute_fbank(waveform,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0,
                  fs=16000,
                  snip_edges=True,
                  window_type="hamming"):
    sample_rate = fs
    waveform = waveform * (1 << 15)
    waveform = waveform.unsqueeze(0)
    # Only keep key, feat, label
    mat = kaldi.fbank(waveform,
                      num_mel_bins=num_mel_bins,
                      frame_length=frame_length,
                      frame_shift=frame_shift,
                      dither=dither,
                      energy_floor=0.0,
                      sample_frequency=sample_rate,
                      window_type=window_type,
                      snip_edges=snip_edges)
    return mat


def compute_w2vbert_fbank(sample,
                          num_mel_bins=23,
                          frame_length=25,
                          frame_shift=10,
                          dither=0.0):
    """ Extract Pretrain w2vbert(4.5M hours) fbank
    """
    sample = compute_fbank(sample, num_mel_bins, frame_length, frame_shift,
                           dither)
    mat = sample['feat']
    std, mean = torch.std_mean(mat, dim=0)
    mat = mat.subtract(mean).divide(std)
    sample['feat'] = mat
    return sample


def apply_lfr(inputs, lfr_m, lfr_n):
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / lfr_n))
    left_padding = inputs[0].repeat((lfr_m - 1) // 2, 1)
    inputs = torch.vstack((left_padding, inputs))
    T = T + (lfr_m - 1) // 2
    for i in range(T_lfr):
        if lfr_m <= T - i * lfr_n:
            LFR_inputs.append((inputs[i * lfr_n : i * lfr_n + lfr_m]).view(1, -1))
        else:  # process last LFR frame
            num_padding = lfr_m - (T - i * lfr_n)
            frame = (inputs[i * lfr_n :]).view(-1)
            for _ in range(num_padding):
                frame = torch.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    LFR_outputs = torch.vstack(LFR_inputs)
    return LFR_outputs.type(torch.float32)


def load_cmvn(cmvn_file):
    with open(cmvn_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == "<AddShift>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                add_shift_line = line_item[3 : (len(line_item) - 1)]
                means_list = list(add_shift_line)
                continue
        elif line_item[0] == "<Rescale>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                rescale_line = line_item[3 : (len(line_item) - 1)]
                vars_list = list(rescale_line)
                continue
    means = np.array(means_list).astype(np.float32)
    vars = np.array(vars_list).astype(np.float32)
    cmvn = np.array([means, vars])
    cmvn = torch.as_tensor(cmvn, dtype=torch.float32)
    return cmvn


def apply_cmvn(inputs, cmvn):  # noqa
    """
    Apply CMVN with mvn data
    """
    device = inputs.device
    dtype = inputs.dtype
    frame, dim = inputs.shape
    means = cmvn[0:1, :dim]
    vars = cmvn[1:2, :dim]
    inputs += means.to(device)
    inputs *= vars.to(device)
    return inputs.type(torch.float32)


def pad(sequence, max_length, padding_idx=0):
    if isinstance(sequence, (int, list, tuple)):
        if len(sequence) < max_length:
            sequence = sequence + [padding_idx] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
    elif isinstance(sequence, torch.Tensor):
        if len(sequence) < max_length:
            sequence = torch.cat(
                (sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
        else:
            sequence = sequence[:max_length]
    elif isinstance(sequence, np.ndarray):
        if len(sequence) < max_length:
            sequence = np.concatenate(
                (sequence, np.full((max_length - len(sequence),) + sequence.shape[1:], padding_idx)))
        else:
            sequence = sequence[:max_length]
    else:
        raise Exception("Type mismatch during padding!")
    return sequence


def padding(sequence, padding_length, padding_idx=0, padding_side="right"):
    if isinstance(sequence, (int, list, tuple)):
        if padding_length >= 0:
            sequence = sequence + [padding_idx] * padding_length
        else:
            sequence = sequence[:padding_length]
    elif isinstance(sequence, torch.Tensor):
        if sequence.ndimension() == 2:
            if padding_length >= 0:
                sequence = torch.nn.functional.pad(sequence, (0, padding_length))
            else:
                sequence = sequence[:, :padding_length]
        else:
            if padding_length >= 0:
                if padding_side == "left":
                    sequence = torch.cat(
                        (torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx), sequence))
                else:
                    sequence = torch.cat(
                        (sequence, torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx)))
            else:
                sequence = sequence[:padding_length]
    elif isinstance(sequence, np.ndarray):
        if padding_length >= 0:
            sequence = np.concatenate(
                (sequence, np.full((padding_length,) + sequence.shape[1:], padding_idx)))
        else:
            sequence = sequence[:padding_length]
    else:
        raise Exception("Type mismatch during padding!")
    return sequence


def process_batch(samples, tokenizer):
    input_prompt_lengths = [s["audio_length"] + s['prompt_length'] for s in samples]  # [120, 48, 82, 42]
    input_answer_lengths = [len(s["input_ids"]) - s["audio_length"] - s['prompt_length'] for s in
                            samples]  # [0, 0, 0, 0]
    
    input_prompt_max_length = max(input_prompt_lengths)
    input_answer_max_length = max(input_answer_lengths)
    
    input_ids = torch.stack([
        padding(
            padding(samples[index]["input_ids"], input_prompt_max_length - input_prompt_lengths[index],
                    tokenizer.pad_token_id, padding_side="left"),
            input_answer_max_length - input_answer_lengths[index], tokenizer.pad_token_id
        ) for index in range(len(samples))
    ])
    
    attention_mask = torch.stack([
        padding(
            padding(samples[index]["attention_mask"], input_prompt_max_length - input_prompt_lengths[index],
                    False, padding_side="left"),
            input_answer_max_length - input_answer_lengths[index], False
        ) for index in range(len(samples))
    ])
    
    audio_mel_reallen = [s['audio_mel'].shape[0] for s in samples]
    audio_mel_max_length = max(audio_mel_reallen)
    audio_mel = torch.stack([pad(s['audio_mel'], audio_mel_max_length, 0)
                             for s in samples])
    audio_mel_post_mask = torch.zeros(len(samples), (
        audio_mel_max_length))  # paraformer
    for line, sample in enumerate(samples):
        audio_mel_post_mask[line, :(sample['audio_mel'].shape[0])] = 1
    audio_mel_reallen = torch.tensor(audio_mel_reallen, dtype=torch.int32)
    modality_mask = torch.zeros_like(attention_mask)
    for index in range(len(samples)):
        padding_left = input_prompt_max_length - input_prompt_lengths[index]
        modality_mask[index, padding_left:padding_left + samples[index]["audio_length"]] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "audio_mel": audio_mel,
        "audio_mel_post_mask": audio_mel_post_mask,
        "modality_mask": modality_mask,
        "audio_mel_reallen": audio_mel_reallen
    }

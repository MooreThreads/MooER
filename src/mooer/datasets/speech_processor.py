import copy
import json
import logging
import random
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import numpy as np
import torch
import torchaudio
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


AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


def url_opener(data):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        url = sample['src']
        try:
            pr = urlparse(url)
            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f'curl -s -L {url}'
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(url))


def tar_file_and_group(data):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{src, stream}]

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        assert 'stream' in sample
        stream = tarfile.open(fileobj=sample['stream'], mode="r|*")
        prev_prefix = None
        example = {}
        valid = True
        try:
            for tarinfo in stream:
                name = tarinfo.name
                pos = name.rfind('.')
                assert pos > 0
                prefix, postfix = name[:pos], name[pos + 1:]
                if prev_prefix is not None and prefix != prev_prefix:
                    example['key'] = prev_prefix
                    if valid:
                        yield example
                    example = {}
                    valid = True
                with stream.extractfile(tarinfo) as file_obj:
                    try:
                        if postfix == 'txt':
                            example['txt'] = file_obj.read().decode('utf8').strip()
                        elif postfix in AUDIO_FORMAT_SETS:
                            waveform, sample_rate = torchaudio.load(file_obj)
                            example['wav'] = waveform
                            example['sample_rate'] = sample_rate
                        else:
                            example[postfix] = file_obj.read()
                    except Exception as ex:
                        valid = False
                        logging.warning('error to parse {}'.format(name))
                prev_prefix = prefix
            if prev_prefix is not None:
                example['key'] = prev_prefix
                yield example
            stream.close()
            if 'process' in sample:
                sample['process'].communicate()
            sample['stream'].close()
        except Exception as e:
            logging.warning(e)
            logging.warning('error to parse {}'.format(sample))


def parse_raw(data):
    """ Parse key/wav/txt from json line

        Args:
            data: Iterable[str], str is a json line has key/wav/txt

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        assert 'key' in obj
        assert 'wav' in obj
        assert 'txt' in obj
        key = obj['key']
        wav_file = obj['wav']
        txt = obj['txt']
        try:
            if 'start' in obj:
                assert 'end' in obj
                sample_rate = torchaudio.backend.sox_io_backend.info(
                    wav_file).sample_rate
                start_frame = int(obj['start'] * sample_rate)
                end_frame = int(obj['end'] * sample_rate)
                waveform, _ = torchaudio.backend.sox_io_backend.load(
                    filepath=wav_file,
                    num_frames=end_frame - start_frame,
                    frame_offset=start_frame)
            else:
                waveform, sample_rate = torchaudio.load(wav_file)
            example = dict(key=key,
                           txt=txt,
                           wav=waveform,
                           sample_rate=sample_rate)
            yield example
        except Exception as ex:
            logging.warning('Failed to read {}'.format(wav_file))


def filter(data,
           max_length=10240,
           min_length=10):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        # sample['wav'] is torch.Tensor, we have 100 frames every second
        num_frames = sample['wav'].size(1) / sample['sample_rate'] * 100
        if num_frames < min_length:
            continue
        if num_frames > max_length:
            continue
        yield sample


def refine_text(data, replaced_table, replace_type, concat_token='<|im_end|>'):
    for sample in data:
        assert 'txt' in sample
        assert 'key' in sample
        uttid = sample['key']
        if replaced_table.get(uttid, None) is None:
            continue
        if replace_type == 'replace':
            sample['txt'] = replaced_table[uttid]
        elif replace_type == 'concat':
            sample['txt'] = sample['txt'] + concat_token + replaced_table[uttid]
        elif replace_type == 'concat_r':
            sample['txt'] = replaced_table[uttid] + concat_token + sample['txt']
        elif replace_type == 'instruction':
            sample['txt'] = {
                'asr': sample['txt'],
                'ast': replaced_table[uttid],
                'asr_ast': sample['txt'] + concat_token + replaced_table[uttid]
            }
        else:
            raise KeyError
        yield sample


def resample(data, resample_rate=16000):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['wav'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        yield sample


def speed_perturb(data, speeds=None):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            speeds(List[float]): optional speed

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    if speeds is None:
        speeds = [0.9, 1.0, 1.1]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        speed = random.choice(speeds)
        if speed != 1.0:
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speed)], ['rate', str(sample_rate)]])
            sample['wav'] = wav
        
        yield sample
        
        
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


def gen_llm_inputs(data, tokenizer, ignore_index=-100, input_type='raw',
                   normalize=True, mel_size=128, prompt_template=None,
                   is_inference=False, autoprocesser=None, is_paraformer=False,
                   cmvn=None, prompt_org="Transcribe speech to text. ", adapter_downsample_rate=5,
                   instruction=False):
    if prompt_template is None:
        prompt_template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    if instruction:
        assert isinstance(prompt_org, dict)
    else:
        prompt = prompt_template.format(prompt_org)
    answer_template = "{}"
    fix_length_audio = -1
    input_type = input_type
    normalize = normalize
    for sample in data:
        audio_raw = sample['wav'][0]
        if instruction:
            target_dict = sample['txt']
            task_list = list(target_dict.keys())
            task_now = random.choice(task_list)
            prompt = prompt_template.format(prompt_org[task_now])
            target = target_dict[task_now].replace('▁', ' ')
        else:
            target = sample['txt'].replace('▁', ' ')
        key = sample['key']
        if autoprocesser is not None:
            audio_mel = autoprocesser(audio_raw, sampling_rate=16000, return_tensors="pt")['input_features'].squeeze(0)
            audio_length = audio_mel.shape[0]  # w2v, downsample 4 has been processed in autoprocesser
            audio_length = audio_length // adapter_downsample_rate
            input_type = "mel"
        elif is_paraformer:
            audio_mel = compute_fbank(waveform=audio_raw)
            audio_mel = apply_lfr(inputs=audio_mel, lfr_m=7, lfr_n=6)
            audio_mel = apply_cmvn(audio_mel, cmvn=cmvn)
            audio_length = audio_mel.shape[0]
            audio_length = audio_length // adapter_downsample_rate
            input_type = "mel"
        elif input_type == "raw":
            if normalize:
                audio_raw = torch.nn.functional.layer_norm(audio_raw, audio_raw.shape)
            audio_length = len(audio_raw) // 320  # ad-hoc for fairseq 320x downsample
            audio_length = audio_length // adapter_downsample_rate
        elif input_type == "mel":
            # NOTE: this is for whisper, you can use compute_fbank to support your encoder
            import whisper
            audio_raw = whisper.pad_or_trim(audio_raw)
            audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=mel_size).permute(1, 0)
            audio_length = (audio_mel.shape[0] + 1) // 2  # ad-hoc for whisper for 2x downsample from mel to feats
            audio_length = audio_length // adapter_downsample_rate
        else:
            raise KeyError
        if fix_length_audio > 0:
            audio_length = fix_length_audio
        audio_pseudo = torch.full((audio_length,), -1)
        prompt_ids = tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)
        
        if is_inference:
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)
            example_ids = torch.cat((audio_pseudo, prompt_ids))  # [audio,prompt]
            example_mask = example_ids.ge(-1)
            yield {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "audio": audio_raw if input_type == "raw" else None,
                "audio_mel": audio_mel if input_type == "mel" else None,
                "audio_length": audio_length,
                "key": key,
                "target": target,
                "prompt_length": prompt_length,
            }
        else:
            answer = answer_template.format(target)
            example = prompt + answer
            example_ids = tokenizer.encode(example)
            example_ids.append(tokenizer.eos_token_id)
            example_ids = torch.tensor(
                example_ids, dtype=torch.int64
            )
            example_ids = torch.cat((audio_pseudo, example_ids))
            labels_ids = copy.deepcopy(example_ids)
            labels_ids[:audio_length + prompt_length] = -1
            example_mask = example_ids.ge(-1)
            label_mask = labels_ids.ge(0)
            example_ids[~example_mask] = 0
            labels_ids[~label_mask] = ignore_index
            
            yield {
                "input_ids": example_ids,
                "labels": labels_ids,
                "attention_mask": example_mask,
                "audio": audio_raw if input_type == "raw" else None,
                "audio_mel": audio_mel if input_type == "mel" else None,
                "audio_length": audio_length,
                "key": key,
                "prompt_length": prompt_length,
            }


def spec_aug(data, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10, max_w=80):
    """ Do spec augmentation
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            max_w: max width of time warp

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)
        # time mask
        for i in range(num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for i in range(num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        sample['feat'] = y
        yield sample


def spec_sub(data, max_t=20, num_t_sub=3):
    """ Do spec substitute
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            max_t: max width of time substitute
            num_t_sub: number of time substitute to apply

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        for i in range(num_t_sub):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            # only substitute the earlier time chosen randomly for current time
            pos = random.randint(0, start)
            y[start:end, :] = x[start - pos:end - pos, :]
        sample['feat'] = y
        yield sample


def shuffle(data, shuffle_size=20000):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500, key='feat'):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{key, feat, label}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{key, feat, label}]
    """
    
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: x[key].size(0))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: x[key].size(0))
    for x in buf:
        yield x


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

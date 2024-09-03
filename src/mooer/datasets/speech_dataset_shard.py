import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset
from transformers import AutoFeatureExtractor

import mooer.datasets.speech_processor as processor
from mooer.utils.data_utils import PROMPT_DICT, PROMPT_TEMPLATE_DICT


def read_lists(list_file, num_epochs=1, shuffle=False):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    lists = lists * num_epochs
    if shuffle:
        random.shuffle(lists)
    return lists


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


class Processor(IterableDataset):
    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:
    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data


class DataList(IterableDataset):
    def __init__(self, lists, shuffle=True, partition=True):
        self.lists = lists
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        for index in indexes:
            # yield dict(src=src)
            data = dict(src=self.lists[index])
            data.update(sampler_info)
            yield data


class SpeechDatasetShard(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_config,
                 normalize=True,
                 mel_size=128,
                 tokenizer=None):
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.IGNORE_INDEX = -100
        self.normalize = normalize
        self.mel_size = mel_size
        self.max_length = dataset_config.get('max_length', 2000)
        self.min_length = dataset_config.get('min_length', 20)
        self.prompt_template_key = dataset_config.get('prompt_template_key', 'qwen')
        self.prompt_template = PROMPT_TEMPLATE_DICT[self.prompt_template_key]
        self.prompt_key = dataset_config.get('prompt_key', 'asr')
        if self.prompt_key == 'instruction':
            self.prompt = PROMPT_DICT
        else:
            self.prompt = PROMPT_DICT[self.prompt_key]
        logging.info(f"Use LLM Type {self.prompt_template_key}, "
                     f"Prompt template {self.prompt_template}, "
                     f"Use task type {self.prompt_key}, "
                     f"Prompt {self.prompt}")
        self.is_inference = dataset_config.get('is_inference', False)
        if (dataset_config.get('w2v_bert_path', None) is not None) and (os.path.exists(dataset_config.w2v_bert_path)):
            self.auto_processer = AutoFeatureExtractor.from_pretrained(dataset_config.w2v_bert_path)
        else:
            self.auto_processer = None
        if (dataset_config.get('cmvn_path', None) is not None) and (os.path.exists(dataset_config.cmvn_path)):
            self.cmvn = load_cmvn(dataset_config.cmvn_path)
        else:
            assert self.dataset_config.encoder_name != 'paraformer', 'paraformer must use cmvn'
            self.cmvn = None
        self.num_epochs = dataset_config.get('num_epochs', 1)
        self.adapter_downsample_rate = dataset_config.get('adapter_downsample_rate', 2)
        self.sort = dataset_config.get('sort', True)
        self.replace_text_table = None
        self.replace_text_path = dataset_config.get('replace_text_path', '')
        self.replace_type = dataset_config.get('replace_type', 'replace')
        if self.prompt_key == 'instruction':
            self.replace_type = 'instruction'
        if os.path.exists(self.replace_text_path):
            logging.info(f"Parsing replaced table {self.replace_text_path}..., Method {self.replace_type}")
            self.replace_text_table = self.parse_txt2dict(self.replace_text_path)
        if self.dataset_config.encoder_name in ['paraformer', 'whisper', 'w2v_bert2.0']:
            self.input_type = 'mel'
        else:
            self.input_type = 'raw'
            
    @classmethod
    def parse_txt2dict(cls, txt_path):
        result = {}
        with open(txt_path, 'r') as r:
            for line in r.readlines():
                line = line.strip()
                if line == '':
                    continue
                line = line.split(maxsplit=1)
                if len(line) != 2:
                    continue
                key, text = line
                result[key] = text
        return result
        
    def dataset(self,
                data_type,
                data_list_file,
                shuffle=True,
                partition=True):
        assert data_type in ['raw', 'shard']
        lists = read_lists(data_list_file, num_epochs=self.num_epochs, shuffle=shuffle)
        dataset = DataList(lists, shuffle=shuffle, partition=partition)
        if data_type == 'shard':
            dataset = Processor(dataset, processor.url_opener)
            dataset = Processor(dataset, processor.tar_file_and_group)
        else:
            dataset = Processor(dataset, processor.parse_raw)
        if not self.is_inference:
            dataset = Processor(dataset, processor.filter,
                                max_length=self.max_length,
                                min_length=self.min_length)
        if self.replace_text_table is not None:
            dataset = Processor(dataset, processor.refine_text,
                                replaced_table=self.replace_text_table,
                                replace_type=self.replace_type)
        dataset = Processor(dataset, processor.gen_llm_inputs,
                            tokenizer=self.tokenizer,
                            ignore_index=self.IGNORE_INDEX,
                            normalize=self.normalize,
                            mel_size=self.mel_size,
                            input_type=self.input_type,
                            is_paraformer=self.dataset_config.encoder_name == 'paraformer',
                            prompt_template=self.prompt_template,
                            is_inference=self.is_inference,
                            autoprocesser=self.auto_processer,
                            cmvn=self.cmvn,
                            prompt_org=self.prompt,
                            adapter_downsample_rate=self.adapter_downsample_rate,
                            instruction=self.prompt_key == 'instruction')
        if not self.is_inference:
            # add shuffle
            dataset = Processor(dataset, processor.shuffle)
        if self.sort:
            dataset = Processor(dataset, processor.sort, sort_size=2000, key='audio'
                if self.dataset_config.encoder_name == 'hubert' else 'audio_mel')
        return dataset

    def pad(self, sequence, max_length, padding_idx=0):
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

    @classmethod
    def padding(cls, sequence, padding_length, padding_idx=0, padding_side="right"):
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

    def collator(self, samples):
        assert samples is not None
        input_prompt_lengths = [s["audio_length"] + s['prompt_length'] for s in samples]  # [120, 48, 82, 42]
        input_answer_lengths = [len(s["input_ids"]) - s["audio_length"] - s['prompt_length'] for s in
                                samples]  # [0, 0, 0, 0]
    
        input_prompt_max_length = max(input_prompt_lengths)
        input_answer_max_length = max(input_answer_lengths)
    
        input_ids = torch.stack([
            self.padding(
                self.padding(samples[index]["input_ids"], input_prompt_max_length - input_prompt_lengths[index],
                             self.tokenizer.pad_token_id, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.tokenizer.pad_token_id
            ) for index in range(len(samples))
        ])
    
        attention_mask = torch.stack([
            self.padding(
                self.padding(samples[index]["attention_mask"], input_prompt_max_length - input_prompt_lengths[index],
                             False, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], False
            ) for index in range(len(samples))
        ])
    
        if self.auto_processer is not None:
            audio_mel_max_length = max([s['audio_mel'].shape[0] for s in samples])
            audio_mel = torch.stack([self.pad(s['audio_mel'], audio_mel_max_length, 0)
                                     for s in samples])
            audio_mel_post_mask = torch.zeros(len(samples), (
                audio_mel_max_length))  # w2v-bert
            for line, sample in enumerate(samples):
                audio_mel_post_mask[line, :(sample['audio_mel'].shape[0])] = 1
        elif self.dataset_config.encoder_name == 'paraformer':
            audio_mel_reallen = [s['audio_mel'].shape[0] for s in samples]
            audio_mel_max_length = max(audio_mel_reallen)
            audio_mel = torch.stack([self.pad(s['audio_mel'], audio_mel_max_length, 0)
                                     for s in samples])
            audio_mel_post_mask = torch.zeros(len(samples), (
                audio_mel_max_length))  # paraformer
            for line, sample in enumerate(samples):
                audio_mel_post_mask[line, :(sample['audio_mel'].shape[0])] = 1
            audio_mel_reallen = torch.tensor(audio_mel_reallen, dtype=torch.int32)
        elif self.dataset_config.encoder_name == 'hubert':
            audio_raw_max_length = max([s['audio'].shape[0] for s in samples])
            audio_raw = torch.stack([self.pad(s['audio'], audio_raw_max_length, 0)
                                     for s in samples])
            audio_mask = torch.zeros(len(samples), audio_raw_max_length)
            for line, sample in enumerate(samples):
                audio_mask[line, :sample['audio'].shape[0]] = 1
        elif self.dataset_config.encoder_name == 'whisper':
            audio_mel_max_length = max([s['audio_mel'].shape[0] for s in samples])
            audio_mel = torch.stack([self.pad(s['audio_mel'], audio_mel_max_length, 0)
                                     for s in samples])
            audio_mel_post_mask = torch.zeros(len(samples), (
                    audio_mel_max_length + 1) // 2)  # ad-hoc for whisper for 2x downsample from mel to feats
            for line, sample in enumerate(samples):
                audio_mel_post_mask[line, :(sample['audio_mel'].shape[0] + 1) // 2] = 1
        modality_mask = torch.zeros_like(attention_mask)
        for index in range(len(samples)):
            padding_left = input_prompt_max_length - input_prompt_lengths[index]
            modality_mask[index, padding_left:padding_left + samples[index]["audio_length"]] = True

        keys = [s['key'] for s in samples]
        if self.is_inference:
            targets = [s['target'] for s in samples]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mask": audio_mask if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
                "modality_mask": modality_mask,
                "keys": keys,
                "targets": targets,
                "audio_mel_reallen": audio_mel_reallen if self.dataset_config.encoder_name == 'paraformer' else None
            }
    
        labels = torch.stack([
            self.padding(
                self.padding(samples[index]['labels'], input_prompt_max_length - input_prompt_lengths[index],
                             self.IGNORE_INDEX, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.IGNORE_INDEX)
            for index in range(len(samples))
        ])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mask": audio_mask if self.input_type == "raw" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
            "modality_mask": modality_mask,
            "audio_mel_reallen": audio_mel_reallen if self.dataset_config.encoder_name == 'paraformer' else None,
            "keys": keys,
        }
    
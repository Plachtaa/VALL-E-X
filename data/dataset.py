# Copyright      2023                           (authors: Feiteng Li)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
modified from lhoste.dataset.speech_synthesis.py
"""

import torch
import math
import h5py
from tokenizers import Tokenizer
from typing import Union, List
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer, models
import json
_pad        = '_'
_punctuation = ',.!?-~…'
_letters = 'NQabdefghijklmnopstuvwxyzɑæʃʑçɯɪɔɛɹðəɫɥɸʊɾʒθβŋɦ⁼ʰ`^#*=ˈˌ→↓↑ '
symbols = [_pad] + list(_punctuation) + list(_letters)

language_dict = {
    'en': 0,
    'zh': 1,
    'ja': 2,
    'ar':3 
}

def seq2phone(tokens: Union[List, np.ndarray]):
    """
    Convert tokenized phoneme ID sequence back to phoneme string
    :param tokens: phoneme tokens
    :return: recovered phoneme sequence
    """
    phones = "".join([symbols[i] for i in tokens])
    return phones

class DynamicBatchSampler(torch.utils.data.Sampler):
    def __init__(self, sampler, num_tokens_fn, num_buckets=100, min_size=0, max_size=1000,
                 max_tokens=None, max_sentences=None, drop_last=False):
        """

        :param sampler:
        :param num_tokens_fn: 根据idx返回样本的长度的函数
        :param num_buckets: 利用桶原理将相似长度的样本放在一个batchsize中，桶的数量
        :param min_size: 最小长度的样本， 小于这个值的样本会被过滤掉。 依据这个值来创建样桶
        :param max_size: 最大长度的样本
        :param max_sentences: batch_size, 但是这里可以通过max_sentences 和 max_tokens 共同控制最终的大小
        """
        super(DynamicBatchSampler, self).__init__(sampler)
        self.sampler = sampler
        self.num_tokens_fn = num_tokens_fn
        self.num_buckets = num_buckets

        self.min_size = min_size
        self.max_size = max_size

        assert max_size <= max_tokens, "max_size should be smaller than max tokens"
        assert max_tokens is not None or max_sentences is not None, \
            "max_tokens and max_sentences should not be null at the same time, please specify one parameter at least"
        self.max_tokens = max_tokens if max_tokens is not None else float('Inf')
        self.max_sentences = max_sentences if max_sentences is not None else float('Inf')
        self.drop_last = drop_last

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
    def is_batch_full(self, num_tokens, batch):
        if len(batch) == 0:
            return False
        if len(batch) == self.max_sentences:
            return True
        if num_tokens > self.max_tokens:
            return True
        return False

    def __iter__(self):
        buckets = [[] for _ in range(self.num_buckets)]
        sample_len = [0] * self.num_buckets

        for idx in self.sampler:
            idx_length = self.num_tokens_fn(idx)
            if not (self.min_size <= idx_length <= self.max_size):
                print("sentence at index {} of size {} exceeds max_tokens, the sentence is ignored".format(idx, idx_length))
                continue

            index_buckets = math.floor((idx_length - self.min_size) / (self.max_size - self.min_size + 1)
                                       * self.num_buckets)
            sample_len[index_buckets] = max(sample_len[index_buckets], idx_length)

            num_tokens = (len(buckets[index_buckets]) + 1) * sample_len[index_buckets]
            if self.is_batch_full(num_tokens, buckets[index_buckets]):
                # yield this batch
                yield buckets[index_buckets]
                buckets[index_buckets] = []
                sample_len[index_buckets] = 0

            buckets[index_buckets].append(idx)

        # process left-over
        leftover_batch = []
        leftover_sample_len = 0
        leftover = [idx for bucket in buckets for idx in bucket]
        for idx in leftover:
            idx_length = self.num_tokens_fn(idx)
            leftover_sample_len = max(leftover_sample_len, idx_length)
            num_tokens = (len(leftover_batch) + 1) * leftover_sample_len
            if self.is_batch_full(num_tokens, leftover_batch):
                yield leftover_batch
                leftover_batch = []
                leftover_sample_len = 0
            leftover_batch.append(idx)

        if len(leftover_batch) > 0 and not self.drop_last:
            yield leftover_batch

    def __len__(self):
        # we do not know the exactly batch size, so do not call len(dataloader)
        pass

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, ann_path, tokenizer_path):
        self.h5_path = h5_path
        with open(ann_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        ls = [l.split("|") for l in lines]
        ls_T = list(zip(*ls))
        self.h5_paths, self.durations, self.langs, self.texts = \
            list(ls_T[0]), list(ls_T[1]), list(ls_T[2]), list(ls_T[3])
        self.durations = [float(dur) for dur in self.durations]
        
        self.tokenizer = None  # Initialize tokenizer attribute to None
        
        try:
            with open(tokenizer_path, "r", encoding="utf-8") as file:
                tokenizer_config = file.read()
                tokenizer_config = json.loads(tokenizer_config)
                bpe_model = models.BPE(
                    vocab=tokenizer_config["model"]["vocab"],
                    merges=tokenizer_config["model"]["merges"]
                )
                self.tokenizer = Tokenizer(bpe_model)
                print("Tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading tokenizer from {tokenizer_path}: {e}")

        self._archive = None

    def __len__(self):
        return len(self.h5_paths)

    def get_dur(self, idx):
        return self.durations[idx]

    @property
    def archive(self):
        if self._archive is None:
            self._archive = h5py.File(self.h5_path, "r")
        return self._archive

    def __getitem__(self, idx):
        archive = self.archive
        h5_path = self.h5_paths[idx]
        sub = archive[h5_path]
        audio_tokens = sub['audio'][()]
        dur = self.durations[idx]
        lang = self.langs[idx]
        text = self.texts[idx]
        print("dur ---> ",dur)
        print("audio--->",audio_tokens)
        # Attempt to access the 'text' dataset, handle KeyError if it doesn't exist
        try:
            phone_tokens = sub['text'][()]
            phones = seq2phone(phone_tokens)
            #print("Phones ---> ",phones)
        except KeyError:
            #print(f"Warning: 'text' dataset not found for index {idx}. Using text directly.")
            phones = text
            #print("Phones ---> ",phones)
        # Tokenize the text
        phones = phones.replace(" ", "_")
        if self.tokenizer is not None:
            if not len(phones):
                cptpho_tokens = self.tokenizer.encode(text).ids
                print("cptpho_tokens ---> ",cptpho_tokens)
            else:
                cptpho_tokens = self.tokenizer.encode(phones).ids
                print("cptpho_tokens ---> ",cptpho_tokens)
        else:
            # If tokenizer is not available, use a placeholder tokenization
            cptpho_tokens = [ord(c) for c in phones]
        print("cptpho_tokens ---> ",cptpho_tokens)
        assert len(cptpho_tokens)

        return {
            'utt_id': h5_path,
            'text': text,
            'audio': None,
            'audio_lens': None,
            'audio_features': audio_tokens,
            'audio_features_lens': len(audio_tokens.T),
            'text_tokens': np.array(cptpho_tokens),
            'text_tokens_lens': len(cptpho_tokens),
            'language': language_dict[lang],
        }



def collate(batch):
    #print(batch)
    utt_id_s = [b.get('utt_id', None) for b in batch]
    #print("=== utt_id_s ===",utt_id_s)
    text_s = [b.get('text', None) for b in batch]
    #print("=== text_s ===",text_s)
    audio_s = [b.get('audio', None) for b in batch]
    #print("=== audio_s ===",audio_s)
    audio_lens_s = [b.get('audio_lens', None) for b in batch]
    #print("=== utt_id_s ===",utt_id_s)
    
    # Filter out None values from audio_features_lens_s before computing maximum
    audio_features_lens_s = [b['audio_features_lens'] for b in batch if b.get('audio_features_lens') is not None]
    
    # Compute maximum audio feature length if audio_features_lens_s is not empty
    max_audio_features_len = max(audio_features_lens_s) if audio_features_lens_s else 0
    
    # Filter out None values from text_tokens_lens_s before computing maximum
    text_tokens_lens_s = [b.get('text_tokens_lens', None) for b in batch if b.get('text_tokens_lens') is not None]
    
    # Compute maximum text tokens length if text_tokens_lens_s is not empty
    max_text_tokens_len = max(text_tokens_lens_s) if text_tokens_lens_s else 0
    
    # Create an empty tensor with maximum audio feature length
    audio_features_s = torch.zeros([len(batch), max_audio_features_len, 8], dtype=torch.int64) - 1  # audio pad with -1

    # Create an empty tensor with maximum text tokens length
    text_tokens_s = torch.zeros([len(batch), max_text_tokens_len], dtype=torch.int64) + 3  # [PAD] token id 3

    # Extract language information from each sample in the batch
    language_s = [b.get('language', None) for b in batch]
    
    # Filter out None values and convert language information to a list of integers
    language_s = [lang for lang in language_s if lang is not None]
    
    # Convert language_s to a tensor only if it's not empty
    if language_s:
        language_tensor = torch.LongTensor(language_s)
    else:
        language_tensor = torch.tensor([])  # Empty tensor if language_s is empty
    
    return {
        'utt_id': utt_id_s,
        'text': text_s,
        'audio': audio_s,
        'audio_lens': audio_lens_s,
        'audio_features': audio_features_s,
        'audio_features_lens': torch.LongTensor(np.array(audio_features_lens_s)),
        'text_tokens': text_tokens_s,
        'text_tokens_lens': torch.LongTensor(np.array(text_tokens_lens_s)),
        'languages': language_tensor,
    }




def create_dataloader(data_dir="/root/valle/egs/mix", n_gpus=1, rank=0, num_workers=0, num_buckets=10, max_duration=120):
    train_dataset = AudioDataset(h5_path=f"{data_dir}/audio_sum.hdf5",
                                 ann_path=f"{data_dir}/audio_ann_sum.txt",
                                 tokenizer_path=f"{data_dir}/bpe_69.json")
    ran_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
        )
    dynamic_sampler = DynamicBatchSampler(ran_sampler, train_dataset.get_dur, num_buckets=num_buckets, max_size=20,
                                          max_tokens=max_duration)


    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=num_workers, collate_fn=collate,
                                               batch_sampler=dynamic_sampler)

    return train_loader

# Copyright      2023                          (authors: Feiteng Li)
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


import argparse
import inspect
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
# from icefall.utils import str2bool
# from lhotse import CutSet, load_manifest_lazy
# from lhotse.dataset import (
#     CutConcatenate,
#     DynamicBucketingSampler,
#     PrecomputedFeatures,
#     SingleCutSampler,
#     SpecAugment,
# )
# from lhotse.dataset.input_strategies import OnTheFlyFeatures
# from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

from data.collation import get_text_token_collater
# from data.dataset import SpeechSynthesisDataset
from data.fbank import get_fbank_extractor
from data.input_strategies import PromptedPrecomputedFeatures

# PrecomputedFeatures = PrecomputedFeatures


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


def _get_input_strategy(input_strategy, dataset, cuts):
    if input_strategy == "PromptedPrecomputedFeatures":
        return PromptedPrecomputedFeatures(dataset, cuts)

    return eval(input_strategy)()


class TtsDataModule:
    """
    DataModule for VALL-E TTS experiments.
    It assumes there is always one train and valid dataloader.

    It contains all the common data pipeline modules used in TTS
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation[not used & tested yet],
    - augmentation[not used & tested yet],
    - on-the-fly feature extraction[not used & tested yet]

    This class should be derived for specific corpora used in TTS tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="TTS data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/tokenized"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=40.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="When enabled, the batches will come from buckets of "
            "similar duration (saves padding frames).",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=10,
            help="The number of buckets for the DynamicBucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=False,
            help="When enabled, utterances (cuts) will be concatenated "
            "to minimize the amount of padding.",
        )
        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
            help="Determines the maximum duration of a concatenated cut "
            "relative to the duration of the longest cut in a batch.",
        )
        group.add_argument(
            "--gap",
            type=float,
            default=0.1,
            help="The amount of padding (in seconds) inserted between "
            "concatenated cuts. This padding is filled with noise when "
            "noise augmentation is used.",
        )
        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=False,
            help="When enabled, use on-the-fly cut mixing and feature "
            "extraction. Will drop existing precomputed feature manifests "
            "if available.",
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="When enabled (=default), the examples will be "
            "shuffled for each epoch.",
        )
        group.add_argument(
            "--drop-last",
            type=str2bool,
            default=False,
            help="Whether to drop last batch. Used by sampler.",
        )
        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
            help="When enabled, each batch will have the "
            "field: batch['supervisions']['cut'] with the cuts that "
            "were used to construct it.",
        )

        group.add_argument(
            "--num-workers",
            type=int,
            default=8,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )

        group.add_argument(
            "--enable-spec-aug",
            type=str2bool,
            default=False,
            help="When enabled, use SpecAugment for training dataset.",
        )

        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help="Used only when --enable-spec-aug is True. "
            "It specifies the factor for time warping in SpecAugment. "
            "Larger values mean more warping. "
            "A value less than 1 means to disable time warp.",
        )

        group.add_argument(
            "--input-strategy",
            type=str,
            default="PrecomputedFeatures",
            help="AudioSamples or PrecomputedFeatures or PromptedPrecomputedFeatures",
        )

        group.add_argument(
            "--dataset",
            type=str,
            default="ljspeech",
            help="--input-strategy PromptedPrecomputedFeatures needs dataset name to prepare prompts.",
        )

        parser.add_argument(
            "--text-tokens",
            type=str,
            default="data/tokenized/unique_text_tokens.k2symbols",
            help="Path to the unique text tokens file",
        )

        parser.add_argument(
            "--sampling-rate",
            type=int,
            default=24000,
            help="""Audio sampling rate.""",
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        transforms = []

        if self.args.concatenate_cuts:
            logging.info(
                f"Using cut concatenation with duration factor "
                f"{self.args.duration_factor} and gap {self.args.gap}."
            )
            # Cut concatenation should be the first transform in the list,
            # so that if we e.g. mix noise in, it will fill the gaps between
            # different utterances.
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        input_transforms = []
        if self.args.enable_spec_aug:
            logging.info("Enable SpecAugment")
            logging.info(
                f"Time warp factor: {self.args.spec_aug_time_warp_factor}"
            )
            # Set the value of num_frame_masks according to Lhotse's version.
            # In different Lhotse's versions, the default of num_frame_masks is
            # different.
            num_frame_masks = 10
            num_frame_masks_parameter = inspect.signature(
                SpecAugment.__init__
            ).parameters["num_frame_masks"]
            if num_frame_masks_parameter.default == 1:
                num_frame_masks = 2
            logging.info(f"Num frame mask: {num_frame_masks}")
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            )
        else:
            logging.info("Disable SpecAugment")

        logging.info("About to create train dataset")
        if self.args.on_the_fly_feats:
            # NOTE: the PerturbSpeed transform should be added only if we
            # remove it from data prep stage.
            # Add on-the-fly speed perturbation; since originally it would
            # have increased epoch size by 3, we will apply prob 2/3 and use
            # 3x more epochs.
            # Speed perturbation probably should come first before
            # concatenation, but in principle the transforms order doesn't have
            # to be strict (e.g. could be randomized)
            # transforms = [PerturbSpeed(factors=[0.9, 1.1], p=2/3)] + transforms   # noqa
            # Drop feats to be on the safe side.
            train = SpeechSynthesisDataset(
                get_text_token_collater(self.args.text_tokens),
                cut_transforms=transforms,
                feature_input_strategy=OnTheFlyFeatures(get_fbank_extractor()),
                feature_transforms=input_transforms,
            )
        else:
            train = SpeechSynthesisDataset(
                get_text_token_collater(self.args.text_tokens),
                feature_input_strategy=_get_input_strategy(
                    self.args.input_strategy, self.args.dataset, cuts_train
                ),
                cut_transforms=transforms,
                feature_transforms=input_transforms,
            )

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler")
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                drop_last=self.args.drop_last,
            )
        else:
            logging.info(
                "Using SingleCutSampler and sort by duraton(ascending=True)."
            )
            cuts_train = cuts_train.to_eager().sort_by_duration(ascending=True)
            train_sampler = SingleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )
        logging.info("About to create train dataloader")

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        # 'seed' is derived from the current random state, which will have
        # previously been set in the main process.
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        logging.info("About to create dev dataset")
        if self.args.on_the_fly_feats:
            validate = SpeechSynthesisDataset(
                get_text_token_collater(self.args.text_tokens),
                feature_input_strategy=OnTheFlyFeatures(get_fbank_extractor()),
                cut_transforms=[],
            )
        else:
            validate = SpeechSynthesisDataset(
                get_text_token_collater(self.args.text_tokens),
                feature_input_strategy=_get_input_strategy(
                    self.args.input_strategy, self.args.dataset, cuts_valid
                ),
                cut_transforms=[],
            )
        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.info("About to create dev dataloader")
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=4,
            persistent_workers=False,
        )

        return valid_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        test = SpeechSynthesisDataset(
            get_text_token_collater(self.args.text_tokens),
            feature_input_strategy=OnTheFlyFeatures(get_fbank_extractor())
            if self.args.on_the_fly_feats
            else _get_input_strategy(
                self.args.input_strategy, self.args.dataset, cuts
            ),
            cut_transforms=[],
        )
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.debug("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl

    @lru_cache()
    def train_cuts(self) -> CutSet:
        logging.info("About to get train cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "cuts_train.jsonl.gz"
        )

    @lru_cache()
    def dev_cuts(self) -> CutSet:
        logging.info("About to get dev cuts")
        return load_manifest_lazy(self.args.manifest_dir / "cuts_dev.jsonl.gz")

    @lru_cache()
    def test_cuts(self) -> CutSet:
        logging.info("About to get test cuts")
        return load_manifest_lazy(self.args.manifest_dir / "cuts_test.jsonl.gz")

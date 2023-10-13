import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Type

from lhotse import CutSet
from lhotse.dataset.collation import collate_features
from lhotse.dataset.input_strategies import (
    ExecutorType,
    PrecomputedFeatures,
    _get_executor,
)
from lhotse.utils import fastcopy


class PromptedFeatures:
    def __init__(self, prompts, features):
        self.prompts = prompts
        self.features = features

    def to(self, device):
        return PromptedFeatures(
            self.prompts.to(device), self.features.to(device)
        )

    def sum(self):
        return self.features.sum()

    @property
    def ndim(self):
        return self.features.ndim

    @property
    def data(self):
        return (self.prompts, self.features)


class PromptedPrecomputedFeatures(PrecomputedFeatures):
    """
    :class:`InputStrategy` that reads pre-computed features, whose manifests
    are attached to cuts, from disk.

    It automatically pads the feature matrices with pre or post feature.

    .. automethod:: __call__
    """

    def __init__(
        self,
        dataset: str,
        cuts: CutSet,
        num_workers: int = 0,
        executor_type: Type[ExecutorType] = ThreadPoolExecutor,
    ) -> None:
        super(PromptedPrecomputedFeatures, self).__init__(
            num_workers, executor_type
        )

        self.utt2neighbors = defaultdict(lambda: [])

        if dataset.lower() == "libritts":
            # 909_131041_000013_000002
            # 909_131041_000013_000003
            speaker2utts = defaultdict(lambda: [])

            utt2cut = {}
            for cut in cuts:
                speaker = cut.supervisions[0].speaker
                speaker2utts[speaker].append(cut.id)
                utt2cut[cut.id] = cut

            for spk in speaker2utts:
                uttids = sorted(speaker2utts[spk])
                # Using the property of sorted keys to find previous utterance
                # The keys has structure speaker_book_x_y e.g. 1089_134691_000004_000001
                if len(uttids) == 1:
                    self.utt2neighbors[uttids[0]].append(utt2cut[uttids[0]])
                    continue

                utt2prevutt = dict(zip(uttids, [uttids[1]] + uttids[:-1]))
                utt2postutt = dict(zip(uttids[:-1], uttids[1:]))

                for utt in utt2prevutt:
                    self.utt2neighbors[utt].append(utt2cut[utt2prevutt[utt]])

                for utt in utt2postutt:
                    self.utt2neighbors[utt].append(utt2cut[utt2postutt[utt]])
        elif dataset.lower() == "ljspeech":
            utt2cut = {}
            uttids = []
            for cut in cuts:
                uttids.append(cut.id)
                utt2cut[cut.id] = cut

            if len(uttids) == 1:
                self.utt2neighbors[uttids[0]].append(utt2cut[uttids[0]])
            else:
                # Using the property of sorted keys to find previous utterance
                # The keys has structure: LJ001-0010
                utt2prevutt = dict(zip(uttids, [uttids[1]] + uttids[:-1]))
                utt2postutt = dict(zip(uttids[:-1], uttids[1:]))

                for utt in utt2postutt:
                    postutt = utt2postutt[utt]
                    if utt[:5] == postutt[:5]:
                        self.utt2neighbors[utt].append(utt2cut[postutt])

                for utt in utt2prevutt:
                    prevutt = utt2prevutt[utt]
                    if utt[:5] == prevutt[:5] or not self.utt2neighbors[utt]:
                        self.utt2neighbors[utt].append(utt2cut[prevutt])
        else:
            raise ValueError

    def __call__(
        self, cuts: CutSet
    ) -> Tuple[PromptedFeatures, PromptedFeatures]:
        """
        Reads the pre-computed features from disk/other storage.
        The returned shape is``(B, T, F) => (batch_size, num_frames, num_features)``.

        :return: a tensor with collated features, and a tensor of ``num_frames`` of each cut before padding.
        """
        features, features_lens = collate_features(
            cuts,
            executor=_get_executor(
                self.num_workers, executor_type=self._executor_type
            ),
        )

        prompts_cuts = []
        for k, cut in enumerate(cuts):
            prompts_cut = random.choice(self.utt2neighbors[cut.id])
            prompts_cuts.append(fastcopy(prompts_cut, id=f"{cut.id}-{str(k)}"))

        mini_duration = min([cut.duration for cut in prompts_cuts] + [3.0])
        # prompts_cuts = CutSet.from_cuts(prompts_cuts).truncate(
        #     max_duration=mini_duration,
        #     offset_type="random",
        #     preserve_id=True,
        # )
        prompts_cuts = CutSet(
            cuts={k: cut for k, cut in enumerate(prompts_cuts)}
        ).truncate(
            max_duration=mini_duration,
            offset_type="random",
            preserve_id=False,
        )

        prompts, prompts_lens = collate_features(
            prompts_cuts,
            executor=_get_executor(
                self.num_workers, executor_type=self._executor_type
            ),
        )

        return PromptedFeatures(prompts, features), PromptedFeatures(
            prompts_lens, features_lens
        )

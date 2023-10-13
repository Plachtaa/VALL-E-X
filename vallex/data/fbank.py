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


from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from lhotse.features.base import FeatureExtractor
from lhotse.utils import EPSILON, Seconds, compute_num_frames
from librosa.filters import mel as librosa_mel_fn


@dataclass
class BigVGANFbankConfig:
    # Spectogram-related part
    # Note that frame_length and frame_shift will be converted to milliseconds before torchaudio/Kaldi sees them
    frame_length: Seconds = 1024 / 24000.0
    frame_shift: Seconds = 256 / 24000.0
    remove_dc_offset: bool = True
    round_to_power_of_two: bool = True

    # Fbank-related part
    low_freq: float = 0.0
    high_freq: float = 12000.0
    num_mel_bins: int = 100
    use_energy: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "BigVGANFbankConfig":
        return BigVGANFbankConfig(**data)


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


# https://github.com/NVIDIA/BigVGAN
# bigvgan_24khz_100band https://drive.google.com/drive/folders/1EpxX6AsxjCbbk0mmAhE0td6eYiABr8Oz
class BigVGANFbank(FeatureExtractor):
    name = "fbank"
    config_type = BigVGANFbankConfig

    def __init__(self, config: Optional[Any] = None):
        super(BigVGANFbank, self).__init__(config)
        sampling_rate = 24000
        self.mel_basis = torch.from_numpy(
            librosa_mel_fn(
                sampling_rate,
                1024,
                self.config.num_mel_bins,
                self.config.low_freq,
                self.config.high_freq,
            ).astype(np.float32)
        )
        self.hann_window = torch.hann_window(1024)

    def _feature_fn(self, samples, **kwargs):
        win_length, n_fft = 1024, 1024
        hop_size = 256
        if True:
            sampling_rate = 24000
            duration = round(samples.shape[-1] / sampling_rate, ndigits=12)
            expected_num_frames = compute_num_frames(
                duration=duration,
                frame_shift=self.frame_shift,
                sampling_rate=sampling_rate,
            )
            pad_size = (
                (expected_num_frames - 1) * hop_size
                + win_length
                - samples.shape[-1]
            )
            assert pad_size >= 0

            y = torch.nn.functional.pad(
                samples,
                (0, pad_size),
                mode="constant",
            )
        else:
            y = torch.nn.functional.pad(
                samples,
                (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                mode="reflect",
            )

        y = y.squeeze(1)

        # complex tensor as default, then use view_as_real for future pytorch compatibility
        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_length,
            window=self.hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        spec = torch.matmul(self.mel_basis, spec)
        spec = spectral_normalize_torch(spec)

        return spec.transpose(2, 1).squeeze(0)

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> np.ndarray:
        assert sampling_rate == 24000
        params = asdict(self.config)
        params.update({"sample_frequency": sampling_rate, "snip_edges": False})
        params["frame_shift"] *= 1000.0
        params["frame_length"] *= 1000.0
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        # Torchaudio Kaldi feature extractors expect the channel dimension to be first.
        if len(samples.shape) == 1:
            samples = samples.unsqueeze(0)
        features = self._feature_fn(samples, **params).to(torch.float32)
        return features.numpy()

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_mel_bins

    @staticmethod
    def mix(
        features_a: np.ndarray,
        features_b: np.ndarray,
        energy_scaling_factor_b: float,
    ) -> np.ndarray:
        return np.log(
            np.maximum(
                # protection against log(0); max with EPSILON is adequate since these are energies (always >= 0)
                EPSILON,
                np.exp(features_a)
                + energy_scaling_factor_b * np.exp(features_b),
            )
        )

    @staticmethod
    def compute_energy(features: np.ndarray) -> float:
        return float(np.sum(np.exp(features)))


def get_fbank_extractor() -> BigVGANFbank:
    return BigVGANFbank(BigVGANFbankConfig())


if __name__ == "__main__":
    extractor = BigVGANFbank(BigVGANFbankConfig())

    samples = torch.from_numpy(np.random.random([1000]).astype(np.float32))
    samples = torch.clip(samples, -1.0, 1.0)
    fbank = extractor.extract(samples, 24000.0)
    print(f"fbank {fbank.shape}")

    from scipy.io.wavfile import read

    MAX_WAV_VALUE = 32768.0

    sampling_rate, samples = read(
        "egs/libritts/prompts/5639_40744_000000_000002.wav"
    )
    print(f"samples: [{samples.min()}, {samples.max()}]")
    fbank = extractor.extract(samples.astype(np.float32) / MAX_WAV_VALUE, 24000)
    print(f"fbank {fbank.shape}")

    import matplotlib.pyplot as plt

    _ = plt.figure(figsize=(18, 10))
    plt.imshow(
        X=fbank.transpose(1, 0),
        cmap=plt.get_cmap("jet"),
        aspect="auto",
        interpolation="nearest",
    )
    plt.gca().invert_yaxis()
    plt.savefig("egs/libritts/prompts/5639_40744_000000_000002.png")
    plt.close()

    print("fbank test PASS!")

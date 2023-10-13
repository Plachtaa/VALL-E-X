#!/usr/bin/env python3
# Copyright    2023                           (authors: Feiteng Li)
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


from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize(
    predicts: Tuple[torch.Tensor],
    batch: Dict[str, Union[List, torch.Tensor]],
    output_dir: str,
    limit: int = 4,
) -> None:
    text_tokens = batch["text_tokens"].to("cpu").detach().numpy()
    text_tokens_lens = batch["text_tokens_lens"].to("cpu").detach().numpy()
    audio_features = batch["audio_features"].to("cpu").detach().numpy()
    audio_features_lens = (
        batch["audio_features_lens"].to("cpu").detach().numpy()
    )
    assert text_tokens.ndim == 2

    utt_ids, texts = batch["utt_id"], batch["text"]

    encoder_outputs = predicts[0].to("cpu").type(torch.float32).detach().numpy()
    decoder_outputs = predicts[1]
    if isinstance(decoder_outputs, list):
        decoder_outputs = decoder_outputs[-1]
    decoder_outputs = (
        decoder_outputs.to("cpu").type(torch.float32).detach().numpy()
    )

    vmin, vmax = 0, 1024  # Encodec
    if decoder_outputs.dtype == np.float32:
        vmin, vmax = -6, 0  # Fbank

    num_figures = 3
    for b, (utt_id, text) in enumerate(zip(utt_ids[:limit], texts[:limit])):
        _ = plt.figure(figsize=(14, 8 * num_figures))

        S = text_tokens_lens[b]
        T = audio_features_lens[b]

        # encoder
        plt.subplot(num_figures, 1, 1)
        plt.title(f"Text: {text}")
        plt.imshow(
            X=np.transpose(encoder_outputs[b]),
            cmap=plt.get_cmap("jet"),
            aspect="auto",
            interpolation="nearest",
        )
        plt.gca().invert_yaxis()
        plt.axvline(x=S - 0.4, linewidth=2, color="r")
        plt.xlabel("Encoder Output")
        plt.colorbar()

        # decoder
        plt.subplot(num_figures, 1, 2)
        plt.imshow(
            X=np.transpose(decoder_outputs[b]),
            cmap=plt.get_cmap("jet"),
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        plt.gca().invert_yaxis()
        plt.axvline(x=T - 0.4, linewidth=2, color="r")
        plt.xlabel("Decoder Output")
        plt.colorbar()

        # target
        plt.subplot(num_figures, 1, 3)
        plt.imshow(
            X=np.transpose(audio_features[b]),
            cmap=plt.get_cmap("jet"),
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        plt.gca().invert_yaxis()
        plt.axvline(x=T - 0.4, linewidth=2, color="r")
        plt.xlabel("Decoder Target")
        plt.colorbar()

        plt.savefig(f"{output_dir}/{utt_id}.png")
        plt.close()

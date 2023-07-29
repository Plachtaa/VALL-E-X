# Copyright    2023                             (authors: Feiteng Li)
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

from functools import partial
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
# from icefall.utils import make_pad_mask
# from torchmetrics.classification import BinaryAccuracy

from models.vallex import Transpose
from modules.embedding import SinePositionalEmbedding, TokenEmbedding
from modules.scaling import BalancedDoubleSwish, ScaledLinear
from modules.transformer import (
    BalancedBasicNorm,
    IdentityNorm,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

from .macros import NUM_MEL_BINS, NUM_TEXT_TOKENS
from .visualizer import visualize

IdentityNorm = IdentityNorm


class Transformer(nn.Module):
    """It implements seq2seq Transformer TTS for debug(No StopPredictor and SpeakerEmbeding)
    Neural Speech Synthesis with Transformer Network
    https://arxiv.org/abs/1809.08895
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        norm_first: bool = True,
        add_prenet: bool = False,
        scaling_xformers: bool = False,
    ):
        """
        Args:
          d_model:
            The number of expected features in the input (required).
          nhead:
            The number of heads in the multiheadattention models (required).
          num_layers:
            The number of sub-decoder-layers in the decoder (required).
        """
        super().__init__()
        self.text_embedding = TokenEmbedding(d_model, NUM_TEXT_TOKENS)  # W_x

        if add_prenet:
            self.encoder_prenet = nn.Sequential(
                Transpose(),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(0.5),
                Transpose(),
                nn.Linear(d_model, d_model),
            )

            self.decoder_prenet = nn.Sequential(
                nn.Linear(NUM_MEL_BINS, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, d_model),
            )

            assert scaling_xformers is False  # TODO: update this block
        else:
            self.encoder_prenet = nn.Identity()
            if scaling_xformers:
                self.decoder_prenet = ScaledLinear(NUM_MEL_BINS, d_model)
            else:
                self.decoder_prenet = nn.Linear(NUM_MEL_BINS, d_model)

        self.encoder_position = SinePositionalEmbedding(
            d_model,
            dropout=0.1,
            scale=False,
        )
        self.decoder_position = SinePositionalEmbedding(
            d_model, dropout=0.1, scale=False
        )

        if scaling_xformers:
            self.encoder = TransformerEncoder(
                TransformerEncoderLayer(
                    d_model,
                    nhead,
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    batch_first=True,
                    norm_first=norm_first,
                    linear1_self_attention_cls=ScaledLinear,
                    linear2_self_attention_cls=partial(
                        ScaledLinear, initial_scale=0.01
                    ),
                    linear1_feedforward_cls=ScaledLinear,
                    linear2_feedforward_cls=partial(
                        ScaledLinear, initial_scale=0.01
                    ),
                    activation=partial(
                        BalancedDoubleSwish,
                        channel_dim=-1,
                        max_abs=10.0,
                        min_prob=0.25,
                    ),
                    layer_norm_cls=IdentityNorm,
                ),
                num_layers=num_layers,
                norm=BalancedBasicNorm(d_model) if norm_first else None,
            )

            self.decoder = nn.TransformerDecoder(
                TransformerDecoderLayer(
                    d_model,
                    nhead,
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    batch_first=True,
                    norm_first=norm_first,
                    linear1_self_attention_cls=ScaledLinear,
                    linear2_self_attention_cls=partial(
                        ScaledLinear, initial_scale=0.01
                    ),
                    linear1_feedforward_cls=ScaledLinear,
                    linear2_feedforward_cls=partial(
                        ScaledLinear, initial_scale=0.01
                    ),
                    activation=partial(
                        BalancedDoubleSwish,
                        channel_dim=-1,
                        max_abs=10.0,
                        min_prob=0.25,
                    ),
                    layer_norm_cls=IdentityNorm,
                ),
                num_layers=num_layers,
                norm=BalancedBasicNorm(d_model) if norm_first else None,
            )

            self.predict_layer = ScaledLinear(d_model, NUM_MEL_BINS)
            self.stop_layer = nn.Linear(d_model, 1)
        else:
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model,
                    nhead,
                    dim_feedforward=d_model * 4,
                    activation=F.relu,
                    dropout=0.1,
                    batch_first=True,
                    norm_first=norm_first,
                ),
                num_layers=num_layers,
                norm=nn.LayerNorm(d_model) if norm_first else None,
            )

            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model,
                    nhead,
                    dim_feedforward=d_model * 4,
                    activation=F.relu,
                    dropout=0.1,
                    batch_first=True,
                    norm_first=norm_first,
                ),
                num_layers=num_layers,
                norm=nn.LayerNorm(d_model) if norm_first else None,
            )

            self.predict_layer = nn.Linear(d_model, NUM_MEL_BINS)
            self.stop_layer = nn.Linear(d_model, 1)

        self.stop_accuracy_metric = BinaryAccuracy(
            threshold=0.5, multidim_average="global"
        )

    #     self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, (nn.Linear)):
    #         module.weight.data.normal_(mean=0.0, std=0.02)
    #         if isinstance(module, nn.Linear) and module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        y_lens: torch.Tensor,
        reduction: str = "sum",
        train_stage: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Args:
          x:
            A 2-D tensor of shape (N, S).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (N, T, 8).
          y_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          train_stage:
            Not used in this model.
        Returns:
          Return the predicted audio code matrix, cross-entropy loss and Top-10 accuracy.
        """
        del train_stage

        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y_lens.ndim == 1, y_lens.shape

        assert torch.all(x_lens > 0)

        # NOTE: x has been padded in TextTokenCollater
        x_mask = make_pad_mask(x_lens).to(x.device)

        x = self.text_embedding(x)
        x = self.encoder_prenet(x)
        x = self.encoder_position(x)
        x = self.encoder(x, src_key_padding_mask=x_mask)

        total_loss, metrics = 0.0, {}

        y_mask = make_pad_mask(y_lens).to(y.device)
        y_mask_float = y_mask.type(torch.float32)
        data_mask = 1.0 - y_mask_float.unsqueeze(-1)

        # Training
        # AR Decoder
        def pad_y(y):
            y = F.pad(y, (0, 0, 1, 0, 0, 0), value=0).detach()
            # inputs, targets
            return y[:, :-1], y[:, 1:]

        y, targets = pad_y(y * data_mask)  # mask padding as zeros

        y_emb = self.decoder_prenet(y)
        y_pos = self.decoder_position(y_emb)

        y_len = y_lens.max()
        tgt_mask = torch.triu(
            torch.ones(y_len, y_len, device=y.device, dtype=torch.bool),
            diagonal=1,
        )
        y_dec = self.decoder(
            y_pos,
            x,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=x_mask,
        )

        predict = self.predict_layer(y_dec)
        # loss
        total_loss = F.mse_loss(predict, targets, reduction=reduction)

        logits = self.stop_layer(y_dec).squeeze(-1)
        stop_loss = F.binary_cross_entropy_with_logits(
            logits,
            y_mask_float.detach(),
            weight=1.0 + y_mask_float.detach() * 4.0,
            reduction=reduction,
        )
        metrics["stop_loss"] = stop_loss.detach()

        stop_accuracy = self.stop_accuracy_metric(
            (torch.sigmoid(logits) >= 0.5).type(torch.int64),
            y_mask.type(torch.int64),
        )
        # icefall MetricsTracker.norm_items()
        metrics["stop_accuracy"] = stop_accuracy.item() * y_lens.sum().type(
            torch.float32
        )

        return ((x, predict), total_loss + 100.0 * stop_loss, metrics)

    def inference(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: Any = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (1, S).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
        Returns:
          Return the predicted audio code matrix and cross-entropy loss.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape

        assert torch.all(x_lens > 0)

        x_mask = make_pad_mask(x_lens).to(x.device)

        x = self.text_embedding(x)
        x = self.encoder_prenet(x)
        x = self.encoder_position(x)
        x = self.encoder(x, src_key_padding_mask=x_mask)

        x_mask = make_pad_mask(x_lens).to(x.device)

        # AR Decoder
        # TODO: Managing decoder steps avoid repetitive computation
        y = torch.zeros(
            [x.shape[0], 1, NUM_MEL_BINS], dtype=torch.float32, device=x.device
        )
        while True:
            y_emb = self.decoder_prenet(y)
            y_pos = self.decoder_position(y_emb)

            tgt_mask = torch.triu(
                torch.ones(
                    y.shape[1], y.shape[1], device=y.device, dtype=torch.bool
                ),
                diagonal=1,
            )

            y_dec = self.decoder(
                y_pos,
                x,
                tgt_mask=tgt_mask,
                memory_mask=None,
                memory_key_padding_mask=x_mask,
            )
            predict = self.predict_layer(y_dec[:, -1:])

            logits = self.stop_layer(y_dec[:, -1:]) > 0  # sigmoid(0.0) = 0.5
            if y.shape[1] > x_lens.max() * 10 or all(logits.cpu().numpy()):
                print(
                    f"TransformerTTS EOS [Text {x_lens[0]} -> Audio {y.shape[1]}]"
                )
                break

            y = torch.concat([y, predict], dim=1)

        return y[:, 1:]

    def visualize(
        self,
        predicts: Tuple[torch.Tensor],
        batch: Dict[str, Union[List, torch.Tensor]],
        output_dir: str,
        limit: int = 4,
    ) -> None:
        visualize(predicts, batch, output_dir, limit=limit)

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


import torch

from modules.optim import Eden


def calc_lr(step, dim_embed, warmup_steps):
    return dim_embed ** (-0.5) * min(
        step ** (-0.5), step * warmup_steps ** (-1.5)
    )


class NoamScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        base_lr: float,
        optimizer: torch.optim.Optimizer,
        dim_embed: int,
        warmup_steps: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:

        self.dim_embed = dim_embed
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        lr = self.base_lr * calc_lr(
            self._step_count, self.dim_embed, self.warmup_steps
        )
        return [lr] * self.num_param_groups

    def set_step(self, step: int):
        self._step_count = step


def get_scheduler(params, optimizer):
    if params.scheduler_name.lower() == "eden":
        scheduler = Eden(optimizer, 5000, 4, warmup_batches=params.warmup_steps)
    elif params.scheduler_name.lower() == "noam":
        scheduler = NoamScheduler(
            params.base_lr,
            optimizer,
            params.decoder_dim,
            warmup_steps=params.warmup_steps,
        )
        # scheduler.set_step(params.start_batch or params.batch_idx_train)
    elif params.scheduler_name.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            params.warmup_steps,
            optimizer,
            eta_min=params.base_lr,
        )
    else:
        raise NotImplementedError(f"{params.scheduler_name}")

    return scheduler

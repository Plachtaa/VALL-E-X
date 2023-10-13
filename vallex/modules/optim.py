# Copyright      2022  Xiaomi Corp.        (authors: Daniel Povey)
#
# See ../LICENSE for clarification regarding multiple authors
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

import contextlib
import logging
import random
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import torch
from lhotse.utils import fix_random_seed
from torch import Tensor
from torch.optim import Optimizer


class BatchedOptimizer(Optimizer):
    """
    This class adds to class Optimizer the capability to optimize parameters in batches:
    it will stack the parameters and their grads for you so the optimizer can work
    on tensors with an extra leading dimension.  This is intended for speed with GPUs,
    as it reduces the number of kernels launched in the optimizer.

    Args:
      params:
    """

    def __init__(self, params, defaults):
        super(BatchedOptimizer, self).__init__(params, defaults)

    @contextlib.contextmanager
    def batched_params(self, param_group, group_params_names):
        """
        This function returns (technically, yields) a list of
          of tuples (p, state), where
        p is a `fake` parameter that is stacked (over axis 0) from real parameters
        that share the same shape, and its gradient is also stacked;
        `state` is the state corresponding to this batch of parameters
        (it will be physically located in the "state" for one of the real
        parameters, the last one that has any particular shape and dtype).

        This function is decorated as a context manager so that it can
        write parameters back to their "real" locations.

        The idea is, instead of doing:
        <code>
          for p in group["params"]:
             state = self.state[p]
             ...
        </code>
        you can do:
        <code>
          with self.batched_params(group["params"]) as batches:
             for p, state, p_names in batches:
                 ...
        </code>

        Args:
          group: a parameter group, which is a list of parameters; should be
                one of self.param_groups.
          group_params_names: name for each parameter in group,
                which is List[str].
        """
        batches = defaultdict(
            list
        )  # `batches` maps from tuple (dtype_as_str,*shape) to list of nn.Parameter
        batches_names = defaultdict(
            list
        )  # `batches` maps from tuple (dtype_as_str,*shape) to list of str

        assert len(param_group) == len(group_params_names)
        for p, named_p in zip(param_group, group_params_names):
            key = (str(p.dtype), *p.shape)
            batches[key].append(p)
            batches_names[key].append(named_p)

        batches_names_keys = list(batches_names.keys())
        sorted_idx = sorted(
            range(len(batches_names)), key=lambda i: batches_names_keys[i]
        )
        batches_names = [
            batches_names[batches_names_keys[idx]] for idx in sorted_idx
        ]
        batches = [batches[batches_names_keys[idx]] for idx in sorted_idx]

        stacked_params_dict = dict()

        # turn batches into a list, in deterministic order.
        # tuples will contain tuples of (stacked_param, state, stacked_params_names),
        # one for each batch in `batches`.
        tuples = []

        for batch, batch_names in zip(batches, batches_names):
            p = batch[0]
            # we arbitrarily store the state in the
            # state corresponding to the 1st parameter in the
            # group.  class Optimizer will take care of saving/loading state.
            state = self.state[p]
            p_stacked = torch.stack(batch)
            grad = torch.stack(
                [
                    torch.zeros_like(p) if p.grad is None else p.grad
                    for p in batch
                ]
            )
            p_stacked.grad = grad
            stacked_params_dict[key] = p_stacked
            tuples.append((p_stacked, state, batch_names))

        yield tuples  # <-- calling code will do the actual optimization here!

        for ((stacked_params, _state, _names), batch) in zip(tuples, batches):
            for i, p in enumerate(batch):  # batch is list of Parameter
                p.copy_(stacked_params[i])


class ScaledAdam(BatchedOptimizer):
    """
     Implements 'Scaled Adam', a variant of Adam where we scale each parameter's update
     proportional to the norm of that parameter; and also learn the scale of the parameter,
     in log space, subject to upper and lower limits (as if we had factored each parameter as
     param = underlying_param * log_scale.exp())


     Args:
          params:  The parameters or param_groups to optimize (like other Optimizer subclasses)
              lr:  The learning rate.  We will typically use a learning rate schedule that starts
                   at 0.03 and decreases over time, i.e. much higher than other common
                   optimizers.
     clipping_scale: (e.g. 2.0)
                   A scale for gradient-clipping: if specified, the normalized gradients
                   over the whole model will be clipped to have 2-norm equal to
                   `clipping_scale` times the median 2-norm over the most recent period
                   of `clipping_update_period` minibatches.  By "normalized gradients",
                   we mean after multiplying by the rms parameter value for this tensor
                   [for non-scalars]; this is appropriate because our update is scaled
                   by this quantity.
            betas: beta1,beta2 are momentum constants for regular momentum, and moving sum-sq grad.
                   Must satisfy 0 < beta <= beta2 < 1.
     scalar_lr_scale: A scaling factor on the learning rate, that we use to update the
                   scale of each parameter tensor and scalar parameters of the mode..
                   If each parameter were decomposed
                   as p * p_scale.exp(), where (p**2).mean().sqrt() == 1.0, scalar_lr_scale
                   would be a the scaling factor on the learning rate of p_scale.
              eps:  A general-purpose epsilon to prevent division by zero
    param_min_rms: Minimum root-mean-square value of parameter tensor, for purposes of
                   learning the scale on the parameters (we'll constrain the rms of each non-scalar
                   parameter tensor to be >= this value)
    param_max_rms: Maximum root-mean-square value of parameter tensor, for purposes of
                   learning the scale on the parameters (we'll constrain the rms of each non-scalar
                   parameter tensor to be <= this value)
       scalar_max: Maximum absolute value for scalar parameters (applicable if your
                   model has any parameters with numel() == 1).
    size_update_period: The periodicity, in steps, with which we update the size (scale)
                   of the parameter tensor.  This is provided to save a little time
                   in the update.
     clipping_update_period: if clipping_scale is specified, this is the period
    """

    def __init__(
        self,
        params,
        lr=3e-02,
        clipping_scale=None,
        betas=(0.9, 0.98),
        scalar_lr_scale=0.1,
        eps=1.0e-08,
        param_min_rms=1.0e-05,
        param_max_rms=3.0,
        scalar_max=10.0,
        size_update_period=4,
        clipping_update_period=100,
        parameters_names=None,
        show_dominant_parameters=True,
    ):

        assert parameters_names is not None, (
            "Please prepare parameters_names,"
            "which is a List[List[str]]. Each List[str] is for a group"
            "and each str is for a parameter"
        )
        defaults = dict(
            lr=lr,
            clipping_scale=clipping_scale,
            betas=betas,
            scalar_lr_scale=scalar_lr_scale,
            eps=eps,
            param_min_rms=param_min_rms,
            param_max_rms=param_max_rms,
            scalar_max=scalar_max,
            size_update_period=size_update_period,
            clipping_update_period=clipping_update_period,
        )

        super(ScaledAdam, self).__init__(params, defaults)
        assert len(self.param_groups) == len(parameters_names)
        self.parameters_names = parameters_names
        self.show_dominant_parameters = show_dominant_parameters

    def __setstate__(self, state):
        super(ScaledAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        batch = True

        for group, group_params_names in zip(
            self.param_groups, self.parameters_names
        ):

            with self.batched_params(
                group["params"], group_params_names
            ) as batches:

                # batches is list of pairs (stacked_param, state).  stacked_param is like
                # a regular parameter, and will have a .grad, but the 1st dim corresponds to
                # a stacking dim, it is not a real dim.

                if (
                    len(batches[0][1]) == 0
                ):  # if len(first state) == 0: not yet initialized
                    clipping_scale = 1
                else:
                    clipping_scale = self._get_clipping_scale(group, batches)

                for p, state, _ in batches:
                    # Perform optimization step.
                    # grad is not going to be None, we handled that when creating the batches.
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError(
                            "ScaledAdam optimizer does not support sparse gradients"
                        )
                    # State initialization
                    if len(state) == 0:
                        self._init_state(group, p, state)

                    self._step_one_batch(group, p, state, clipping_scale)

        return loss

    def _init_state(self, group: dict, p: Tensor, state: dict):
        """
        Initializes state dict for parameter 'p'.  Assumes that dim 0 of tensor p
        is actually the batch dimension, corresponding to batched-together
        parameters of a given shape.


        Args:
           group:   Dict to look up configuration values.
               p: The parameter that we are initializing the state for
           state: Dict from string to whatever state we are initializing
        """
        size_update_period = group["size_update_period"]

        state["step"] = 0

        kwargs = {"device": p.device, "dtype": p.dtype}

        # 'delta' implements conventional momentum.  There are
        # several different kinds of update going on, so rather than
        # compute "exp_avg" like in Adam, we store and decay a
        # parameter-change "delta", which combines all forms of
        # update.  this is equivalent to how it's done in Adam,
        # except for the first few steps.
        state["delta"] = torch.zeros_like(
            p, memory_format=torch.preserve_format
        )

        batch_size = p.shape[0]
        numel = p.numel() // batch_size
        numel = p.numel()

        if numel > 1:
            # "param_rms" just periodically records the scalar root-mean-square value of
            # the parameter tensor.
            # it has a shape like (batch_size, 1, 1, 1, 1)
            param_rms = (
                (p ** 2).mean(dim=list(range(1, p.ndim)), keepdim=True).sqrt()
            )
            state["param_rms"] = param_rms

            state["scale_exp_avg_sq"] = torch.zeros_like(param_rms)
            state["scale_grads"] = torch.zeros(
                size_update_period, *param_rms.shape, **kwargs
            )

        # exp_avg_sq is the weighted sum of scaled gradients. as in Adam.
        state["exp_avg_sq"] = torch.zeros_like(
            p, memory_format=torch.preserve_format
        )

    def _get_clipping_scale(
        self, group: dict, tuples: List[Tuple[Tensor, dict, List[str]]]
    ) -> float:
        """
        Returns a scalar factor <= 1.0 that dictates gradient clipping, i.e. we will scale the gradients
        by this amount before applying the rest of the update.

        Args:
           group: the parameter group, an item in self.param_groups
           tuples: a list of tuples of (param, state, param_names)
                where param is a batched set of parameters,
                with a .grad (1st dim is batch dim)
                and state is the state-dict where optimization parameters are kept.
                param_names is a List[str] while each str is name for a parameter
                in batched set of parameters "param".
        """
        assert len(tuples) >= 1
        clipping_scale = group["clipping_scale"]
        (first_p, first_state, _) = tuples[0]
        step = first_state["step"]
        if clipping_scale is None or step == 0:
            # no clipping.  return early on step == 0 because the other
            # parameters' state won't have been initialized yet.
            return 1.0
        clipping_update_period = group["clipping_update_period"]

        tot_sumsq = torch.tensor(0.0, device=first_p.device)
        for (p, state, param_names) in tuples:
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError(
                    "ScaledAdam optimizer does not support sparse gradients"
                )
            if p.numel() == p.shape[0]:  # a batch of scalars
                tot_sumsq += (
                    grad ** 2
                ).sum()  # sum() to change shape [1] to []
            else:
                tot_sumsq += ((grad * state["param_rms"]) ** 2).sum()

        tot_norm = tot_sumsq.sqrt()
        if "model_norms" not in first_state:
            first_state["model_norms"] = torch.zeros(
                clipping_update_period, device=p.device
            )
        first_state["model_norms"][step % clipping_update_period] = tot_norm

        if step % clipping_update_period == 0:
            # Print some stats.
            # We don't reach here if step == 0 because we would have returned
            # above.
            sorted_norms = first_state["model_norms"].sort()[0].to("cpu")
            quartiles = []
            for n in range(0, 5):
                index = min(
                    clipping_update_period - 1,
                    (clipping_update_period // 4) * n,
                )
                quartiles.append(sorted_norms[index].item())

            median = quartiles[2]
            threshold = clipping_scale * median
            first_state["model_norm_threshold"] = threshold
            percent_clipped = (
                first_state["num_clipped"] * 100.0 / clipping_update_period
                if "num_clipped" in first_state
                else 0.0
            )
            first_state["num_clipped"] = 0
            quartiles = " ".join(["%.3e" % x for x in quartiles])
            logging.info(
                f"Clipping_scale={clipping_scale}, grad-norm quartiles {quartiles}, "
                f"threshold={threshold:.3e}, percent-clipped={percent_clipped:.1f}"
            )

        if step < clipping_update_period:
            return 1.0  # We have not yet estimated a norm to clip to.
        else:
            try:
                model_norm_threshold = first_state["model_norm_threshold"]
            except KeyError:
                logging.info(
                    "Warning: model_norm_threshold not in state: possibly "
                    "you changed config when restarting, adding clipping_scale option?"
                )
                return 1.0
            ans = min(1.0, (model_norm_threshold / (tot_norm + 1.0e-20)).item())
            if ans < 1.0:
                first_state["num_clipped"] += 1
            if ans < 0.1:
                logging.warn(
                    f"Scaling gradients by {ans}, model_norm_threshold={model_norm_threshold}"
                )
                if self.show_dominant_parameters:
                    assert p.shape[0] == len(param_names)
                    self._show_gradient_dominating_parameter(tuples, tot_sumsq)
            return ans

    def _show_gradient_dominating_parameter(
        self, tuples: List[Tuple[Tensor, dict, List[str]]], tot_sumsq: Tensor
    ):
        """
        Show information of parameter wihch dominanting tot_sumsq.

        Args:
           tuples: a list of tuples of (param, state, param_names)
                where param is a batched set of parameters,
                with a .grad (1st dim is batch dim)
                and state is the state-dict where optimization parameters are kept.
                param_names is a List[str] while each str is name for a parameter
                in batched set of parameters "param".
            tot_sumsq: sumsq of all parameters. Though it's could be calculated
                from tuples, we still pass it to save some time.
        """
        all_sumsq_orig = {}
        for (p, state, batch_param_names) in tuples:
            # p is a stacked batch parameters.
            batch_grad = p.grad
            if p.numel() == p.shape[0]:  # a batch of scalars
                batch_sumsq_orig = batch_grad ** 2
                # Dummpy values used by following `zip` statement.
                batch_rms_orig = torch.ones(p.shape[0])
            else:
                batch_rms_orig = state["param_rms"]
                batch_sumsq_orig = ((batch_grad * batch_rms_orig) ** 2).sum(
                    dim=list(range(1, batch_grad.ndim))
                )

            for name, sumsq_orig, rms, grad in zip(
                batch_param_names, batch_sumsq_orig, batch_rms_orig, batch_grad
            ):

                proportion_orig = sumsq_orig / tot_sumsq
                all_sumsq_orig[name] = (proportion_orig, sumsq_orig, rms, grad)

        assert torch.isclose(
            sum([value[0] for value in all_sumsq_orig.values()]).cpu(),
            torch.tensor(1.0),
        )
        sorted_by_proportion = {
            k: v
            for k, v in sorted(
                all_sumsq_orig.items(),
                key=lambda item: item[1][0],
                reverse=True,
            )
        }
        dominant_param_name = next(iter(sorted_by_proportion))
        (
            dominant_proportion,
            dominant_sumsq,
            dominant_rms,
            dominant_grad,
        ) = sorted_by_proportion[dominant_param_name]
        logging.info(
            f"Parameter Dominanting tot_sumsq {dominant_param_name}"
            f" with proportion {dominant_proportion:.2f},"
            f" where dominant_sumsq=(grad_sumsq*orig_rms_sq)"
            f"={dominant_sumsq:.3e},"
            f" grad_sumsq = {(dominant_grad**2).sum():.3e},"
            f" orig_rms_sq={(dominant_rms**2).item():.3e}"
        )

    def _step_one_batch(
        self, group: dict, p: Tensor, state: dict, clipping_scale: float
    ):
        """
        Do the step for one parameter, which is actually going to be a batch of
        `real` parameters, with dim 0 as the batch dim.
        Args:
                  group:  dict to look up configuration values
                    p: parameter to update (actually multiple parameters stacked together
                       as a batch)
                  state: state-dict for p, to look up the optimizer state
        """
        lr = group["lr"]
        size_update_period = group["size_update_period"]
        beta1 = group["betas"][0]

        grad = p.grad
        if clipping_scale != 1.0:
            grad = grad * clipping_scale
        step = state["step"]
        delta = state["delta"]

        delta.mul_(beta1)
        batch_size = p.shape[0]
        numel = p.numel() // batch_size
        if numel > 1:
            # Update the size/scale of p, and set param_rms
            scale_grads = state["scale_grads"]
            scale_grads[step % size_update_period] = (p * grad).sum(
                dim=list(range(1, p.ndim)), keepdim=True
            )
            if step % size_update_period == size_update_period - 1:
                param_rms = state["param_rms"]  # shape: (batch_size, 1, 1, ..)
                param_rms.copy_(
                    (p ** 2)
                    .mean(dim=list(range(1, p.ndim)), keepdim=True)
                    .sqrt()
                )
                if step > 0:
                    # self._size_update() learns the overall scale on the
                    # parameter, by shrinking or expanding it.
                    self._size_update(group, scale_grads, p, state)

        if numel == 1:
            # For parameters with 1 element we just use regular Adam.
            # Updates delta.
            self._step_scalar(group, p, state)
        else:
            self._step(group, p, state)

        state["step"] = step + 1

    def _size_update(
        self, group: dict, scale_grads: Tensor, p: Tensor, state: dict
    ) -> None:
        """
               Called only where p.numel() > 1, this updates the scale of the parameter.
               If we imagine: p =  underlying_param * scale.exp(), and we are doing
               gradient descent on underlying param and on scale, this function does the update
               on `scale`.

               Args:
              group: dict to look up configuration values
        scale_grads: a tensor of shape (size_update_period, batch_size, 1, 1,...) containing
                      grads w.r.t. the scales.
                  p:  The parameter to update
               state: The state-dict of p
        """

        param_rms = state["param_rms"]
        beta1, beta2 = group["betas"]
        size_lr = group["lr"] * group["scalar_lr_scale"]
        param_min_rms = group["param_min_rms"]
        param_max_rms = group["param_max_rms"]
        eps = group["eps"]
        step = state["step"]
        batch_size = p.shape[0]

        size_update_period = scale_grads.shape[0]
        # correct beta2 for the size update period: we will have
        # faster decay at this level.
        beta2_corr = beta2 ** size_update_period

        scale_exp_avg_sq = state[
            "scale_exp_avg_sq"
        ]  # shape: (batch_size, 1, 1, ..)
        scale_exp_avg_sq.mul_(beta2_corr).add_(
            (scale_grads ** 2).mean(
                dim=0
            ),  # mean over dim `size_update_period`
            alpha=1 - beta2_corr,
        )  # shape is (batch_size, 1, 1, ...)

        # The 1st time we reach here is when size_step == 1.
        size_step = (step + 1) // size_update_period
        bias_correction2 = 1 - beta2_corr ** size_step
        # we don't bother with bias_correction1; this will help prevent divergence
        # at the start of training.

        denom = scale_exp_avg_sq.sqrt() + eps

        scale_step = (
            -size_lr
            * (bias_correction2 ** 0.5)
            * scale_grads.sum(dim=0)
            / denom
        )

        is_too_small = param_rms < param_min_rms
        is_too_large = param_rms > param_max_rms

        # when the param gets too small, just don't shrink it any further.
        scale_step.masked_fill_(is_too_small, 0.0)
        # when it gets too large, stop it from getting any larger.
        scale_step.masked_fill_(is_too_large, -size_lr * size_update_period)
        delta = state["delta"]
        # the factor of (1-beta1) relates to momentum.
        delta.add_(p * scale_step, alpha=(1 - beta1))

    def _step(self, group: dict, p: Tensor, state: dict):
        """
        This function does the core update of self.step(), in the case where the members of
        the batch have more than 1 element.

        Args:
            group: A dict which will be used to look up configuration values
                p: The parameter to be updated
             grad: The grad of p
            state: The state-dict corresponding to parameter p

        This function modifies p.
        """
        grad = p.grad
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        param_min_rms = group["param_min_rms"]
        step = state["step"]

        exp_avg_sq = state["exp_avg_sq"]
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

        this_step = state["step"] - (
            state["zero_step"] if "zero_step" in state else 0
        )
        bias_correction2 = 1 - beta2 ** (this_step + 1)
        if bias_correction2 < 0.99:
            # note: not in-place.
            exp_avg_sq = exp_avg_sq * (1.0 / bias_correction2)

        denom = exp_avg_sq.sqrt()
        denom += eps
        grad = grad / denom

        alpha = -lr * (1 - beta1) * state["param_rms"].clamp(min=param_min_rms)

        delta = state["delta"]
        delta.add_(grad * alpha)
        p.add_(delta)

    def _step_scalar(self, group: dict, p: Tensor, state: dict):
        """
        A simplified form of the core update for scalar tensors, where we cannot get a good
        estimate of the parameter rms.
        """
        beta1, beta2 = group["betas"]
        scalar_max = group["scalar_max"]
        eps = group["eps"]
        lr = group["lr"] * group["scalar_lr_scale"]
        grad = p.grad

        exp_avg_sq = state["exp_avg_sq"]  # shape: (batch_size,)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # bias_correction2 is like in Adam.  Don't bother with bias_correction1;
        # slower update at the start will help stability anyway.
        bias_correction2 = 1 - beta2 ** (state["step"] + 1)
        denom = (exp_avg_sq / bias_correction2).sqrt() + eps

        delta = state["delta"]
        delta.add_(grad / denom, alpha=-lr * (1 - beta1))
        p.clamp_(min=-scalar_max, max=scalar_max)
        p.add_(delta)


class LRScheduler(object):
    """
    Base-class for learning rate schedulers where the learning-rate depends on both the
    batch and the epoch.
    """

    def __init__(self, optimizer: Optimizer, verbose: bool = False):
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(
                "{} is not an Optimizer".format(type(optimizer).__name__)
            )
        self.optimizer = optimizer
        self.verbose = verbose

        for group in optimizer.param_groups:
            group.setdefault("base_lr", group["lr"])

        self.base_lrs = [group["base_lr"] for group in optimizer.param_groups]

        self.epoch = 0
        self.batch = 0

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            "base_lrs": self.base_lrs,
            "epoch": self.epoch,
            "batch": self.batch,
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self) -> List[float]:
        """Return last computed learning rate by current scheduler.  Will be a list of float."""
        return self._last_lr

    def get_lr(self):
        # Compute list of learning rates from self.epoch and self.batch and
        # self.base_lrs; this must be overloaded by the user.
        # e.g. return [some_formula(self.batch, self.epoch, base_lr) for base_lr in self.base_lrs ]
        raise NotImplementedError

    def step_batch(self, batch: Optional[int] = None) -> None:
        # Step the batch index, or just set it.  If `batch` is specified, it
        # must be the batch index from the start of training, i.e. summed over
        # all epochs.
        # You can call this in any order; if you don't provide 'batch', it should
        # of course be called once per batch.
        if batch is not None:
            self.batch = batch
        else:
            self.batch = self.batch + 1
        self._set_lrs()

    def step_epoch(self, epoch: Optional[int] = None):
        # Step the epoch index, or just set it.  If you provide the 'epoch' arg,
        # you should call this at the start of the epoch; if you don't provide the 'epoch'
        # arg, you should call it at the end of the epoch.
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch = self.epoch + 1
        self._set_lrs()

    def _set_lrs(self):
        values = self.get_lr()
        assert len(values) == len(self.optimizer.param_groups)

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group["lr"] = lr
            self.print_lr(self.verbose, i, lr)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def print_lr(self, is_verbose, group, lr):
        """Display the current learning rate."""
        if is_verbose:
            logging.info(
                f"Epoch={self.epoch}, batch={self.batch}: adjusting learning rate"
                f" of group {group} to {lr:.4e}."
            )


class Eden(LRScheduler):
    """
    Eden scheduler.
    The basic formula (before warmup) is:
      lr = base_lr * (((batch**2 + lr_batches**2) / lr_batches**2) ** -0.25 *
                     (((epoch**2 + lr_epochs**2) / lr_epochs**2) ** -0.25)) * warmup
    where `warmup` increases from linearly 0.5 to 1 over `warmup_batches` batches
    and then stays constant at 1.


     E.g. suggest base_lr = 0.04 (passed to optimizer) if used with ScaledAdam

    Args:
        optimizer: the optimizer to change the learning rates on
        lr_batches: the number of batches after which we start significantly
              decreasing the learning rate, suggest 5000.
        lr_epochs: the number of epochs after which we start significantly
              decreasing the learning rate, suggest 6 if you plan to do e.g.
              20 to 40 epochs, but may need smaller number if dataset is huge
              and you will do few epochs.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_batches: Union[int, float],
        lr_epochs: Union[int, float],
        warmup_batches: Union[int, float] = 500.0,
        verbose: bool = False,
    ):
        super(Eden, self).__init__(optimizer, verbose)
        self.lr_batches = lr_batches
        self.lr_epochs = lr_epochs
        self.warmup_batches = warmup_batches

    def get_lr(self):
        factor = (
            (self.batch ** 2 + self.lr_batches ** 2) / self.lr_batches ** 2
        ) ** -0.25 * (
            ((self.epoch ** 2 + self.lr_epochs ** 2) / self.lr_epochs ** 2)
            ** -0.25
        )
        warmup_factor = (
            1.0
            if self.batch >= self.warmup_batches
            else 0.5 + 0.5 * (self.batch / self.warmup_batches)
        )

        return [x * factor * warmup_factor for x in self.base_lrs]


def _test_eden():
    m = torch.nn.Linear(100, 100)
    optim = ScaledAdam(m.parameters(), lr=0.03)

    scheduler = Eden(optim, lr_batches=100, lr_epochs=2, verbose=True)

    for epoch in range(10):
        scheduler.step_epoch(epoch)  # sets epoch to `epoch`

        for step in range(20):
            x = torch.randn(200, 100).detach()
            x.requires_grad = True
            y = m(x)
            dy = torch.randn(200, 100).detach()
            f = (y * dy).sum()
            f.backward()

            optim.step()
            scheduler.step_batch()
            optim.zero_grad()

    logging.info(f"last lr = {scheduler.get_last_lr()}")
    logging.info(f"state dict = {scheduler.state_dict()}")


# This is included mostly as a baseline for ScaledAdam.
class Eve(Optimizer):
    """
    Implements Eve algorithm.  This is a modified version of AdamW with a special
    way of setting the weight-decay / shrinkage-factor, which is designed to make the
    rms of the parameters approach a particular target_rms (default: 0.1).  This is
    for use with networks with 'scaled' versions of modules (see scaling.py), which
    will be close to invariant to the absolute scale on the parameter matrix.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.
    Eve is unpublished so far.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 3e-4;
            this value means that the weight would decay significantly after
            about 3k minibatches.  Is not multiplied by learning rate, but
            is conditional on RMS-value of parameter being > target_rms.
        target_rms (float, optional): target root-mean-square value of
           parameters, if they fall below this we will stop applying weight decay.


    .. _Adam: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=1e-3,
        target_rms=0.1,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )
        if not 0 <= weight_decay <= 0.1:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        if not 0 < target_rms <= 10.0:
            raise ValueError("Invalid target_rms value: {}".format(target_rms))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            target_rms=target_rms,
        )
        super(Eve, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Eve, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() * (bias_correction2 ** -0.5)).add_(
                    group["eps"]
                )

                step_size = group["lr"] / bias_correction1
                target_rms = group["target_rms"]
                weight_decay = group["weight_decay"]

                if p.numel() > 1:
                    # avoid applying this weight-decay on "scaling factors"
                    # (which are scalar).
                    is_above_target_rms = p.norm() > (
                        target_rms * (p.numel() ** 0.5)
                    )
                    p.mul_(1 - (weight_decay * is_above_target_rms))

                p.addcdiv_(exp_avg, denom, value=-step_size)

                # if random.random() < 0.0005:
                #     step = (exp_avg / denom) * step_size
                #     logging.info(
                #         f"Delta rms = {(step**2).mean().item()}, shape = {step.shape}"
                #     )

        return loss


def _test_scaled_adam(hidden_dim: int):
    import timeit

    from scaling import ScaledLinear

    E = 100
    B = 4
    T = 2
    logging.info("in test_eve_cain")
    # device = torch.device('cuda')
    device = torch.device("cpu")
    dtype = torch.float32

    fix_random_seed(42)
    # these input_magnitudes and output_magnitudes are to test that
    # Abel is working as we expect and is able to adjust scales of
    # different dims differently.
    input_magnitudes = (1.0 * torch.randn(E, dtype=dtype, device=device)).exp()
    output_magnitudes = (1.0 * torch.randn(E, dtype=dtype, device=device)).exp()

    for iter in [1, 0]:
        fix_random_seed(42)
        Linear = torch.nn.Linear if iter == 0 else ScaledLinear

        m = torch.nn.Sequential(
            Linear(E, hidden_dim),
            torch.nn.PReLU(),
            Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU(),
            Linear(hidden_dim, E),
        ).to(device)

        train_pairs = [
            (
                100.0
                * torch.randn(B, T, E, device=device, dtype=dtype)
                * input_magnitudes,
                torch.randn(B, T, E, device=device, dtype=dtype)
                * output_magnitudes,
            )
            for _ in range(20)
        ]

        if iter == 0:
            optim = Eve(m.parameters(), lr=0.003)
        elif iter == 1:
            optim = ScaledAdam(m.parameters(), lr=0.03, clipping_scale=2.0)
        scheduler = Eden(optim, lr_batches=200, lr_epochs=5, verbose=False)

        start = timeit.default_timer()
        avg_loss = 0.0
        for epoch in range(180):
            scheduler.step_epoch()
            # if epoch == 100 and iter in [2,3]:
            #    optim.reset_speedup()  # check it doesn't crash.

            # if epoch == 130:
            #    opts = diagnostics.TensorDiagnosticOptions(
            #        2 ** 22
            #    )  # allow 4 megabytes per sub-module
            #    diagnostic = diagnostics.attach_diagnostics(m, opts)

            for n, (x, y) in enumerate(train_pairs):
                y_out = m(x)
                loss = ((y_out - y) ** 2).mean() * 100.0
                if epoch == 0 and n == 0:
                    avg_loss = loss.item()
                else:
                    avg_loss = 0.98 * avg_loss + 0.02 * loss.item()
                if n == 0 and epoch % 5 == 0:
                    # norm1 = '%.2e' % (m[0].weight**2).mean().sqrt().item()
                    # norm1b = '%.2e' % (m[0].bias**2).mean().sqrt().item()
                    # norm2 = '%.2e' % (m[2].weight**2).mean().sqrt().item()
                    # norm2b = '%.2e' % (m[2].bias**2).mean().sqrt().item()
                    # scale1 = '%.2e' % (m[0].weight_scale.exp().item())
                    # scale1b = '%.2e' % (m[0].bias_scale.exp().item())
                    # scale2 = '%.2e' % (m[2].weight_scale.exp().item())
                    # scale2b = '%.2e' % (m[2].bias_scale.exp().item())
                    lr = scheduler.get_last_lr()[0]
                    logging.info(
                        f"Iter {iter}, epoch {epoch}, batch {n}, avg_loss {avg_loss:.4g}, lr={lr:.4e}"
                    )  # , norms={norm1,norm1b,norm2,norm2b}") # scales={scale1,scale1b,scale2,scale2b}
                loss.log().backward()
                optim.step()
                optim.zero_grad()
                scheduler.step_batch()

        # diagnostic.print_diagnostics()

        stop = timeit.default_timer()
        logging.info(f"Iter={iter}, Time taken: {stop - start}")

        logging.info(f"last lr = {scheduler.get_last_lr()}")
        # logging.info("state dict = ", scheduler.state_dict())
        # logging.info("optim state_dict = ", optim.state_dict())
        logging.info(f"input_magnitudes = {input_magnitudes}")
        logging.info(f"output_magnitudes = {output_magnitudes}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    logging.getLogger().setLevel(logging.INFO)
    import subprocess

    s = subprocess.check_output(
        "git status -uno .; git log -1; git diff HEAD .", shell=True
    )
    logging.info(s)
    import sys

    if len(sys.argv) > 1:
        hidden_dim = int(sys.argv[1])
    else:
        hidden_dim = 200

    _test_scaled_adam(hidden_dim)
    _test_eden()

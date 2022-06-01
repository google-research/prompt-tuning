# Copyright 2022 Google.
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

"""IA^3 (https://arxiv.org/abs/2205.05638) implementation."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic

import functools
import operator
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from aqt.jax_legacy.jax import flax_layers as aqt_flax_layers
from aqt.jax_legacy.jax import quant_config as aqt_config
from aqt.jax_legacy.jax import quantization as aqt

import flax.linen as nn
from flax.linen import partitioning
from jax import lax
import jax.numpy as jnp
from prompt_tuning import prompts
from flaxformer import activation_partitioning
from flaxformer.components import dense as ff_dense
from flaxformer.types import Array
from flaxformer.types import DType


class IA3(nn.Module):
  """IA3 scaling to use with a Linear layer.

  Attributes:
    init: How to initialize the scaling variable.
    axis_name: The logical names of the variable axes, used for partitioning.
    dtype: The dtype of the activations for this module.
  """
  ia3_init: prompts.Initializer = nn.initializers.ones
  axis_name: Tuple[str] = ('embed',)
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, x, *args, **kwargs):
    del args
    del kwargs
    *rest, hidden = x.shape
    scaling = partitioning.param_with_axes(
        'ia3_scaling',
        self.ia3_init,
        (hidden,),
        axes=self.axis_name
    )
    scaling = scaling.astype(self.dtype)
    # Reshape to broadcast over batch, seq, etc.
    scaling = jnp.reshape(scaling, tuple((1 for _ in rest)) + scaling.shape)
    return x * scaling


class IA3Attention(nn.Module):
  """A version of IA3 scaling to use with the Attention class.

  Note:
    Because of where we can hook into the flaxformer attention class (the
    `(k|v)_conv` module) the input to this function is already reshaped into
    [..., length, heads, kv] so we shape our scaling to match those last two
    dimensions. This will result in the same value as if we were to reshape
    the variable and do a single d_model scale.
    TODO: Rewrite as a single class that infers the number of dims
    to extract from the input to use to shape the param from the number of dims
    in the axis names.

  Attributes:
    init: How to initialize the scaling variable.
    axis_name: The logical names of the variable axes, used for partitioning.
    dtype: The dtype of the activations for this module.
  """
  ia3_init: prompts.Initializer = nn.initializers.ones
  axis_names: Tuple[str, str] = ('heads', 'kv')
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, x, *args, **kwargs):
    del args
    del kwargs
    *rest, heads, kv = x.shape
    scaling = partitioning.param_with_axes(
        'ia3_scaling',
        self.ia3_init,
        (heads, kv),
        axes=self.axis_names
    )
    scaling = scaling.astype(self.dtype)
    # Reshape to broadcast over batch, seq, etc.
    scaling = jnp.reshape(scaling, tuple((1 for _ in rest)) + scaling.shape)
    return x * scaling


# Aliases to minimize changes in the fork.
# pylint: disable=protected-access
_get_logical_axes = ff_dense._get_logical_axes
_convert_to_activation_function = ff_dense._convert_to_activation_function
# pylint: enable=protected-access
DenseGeneral = ff_dense.DenseGeneral


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Note:
    fork of flaxformer.componenets.dense as there wasn't a good place to hook
    the ia3 transform into the computation.


  Attributes:
    use_bias: Whether to use bias in the dense layers.
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in flax.linen, or a function.
    kernel_init: Kernel function, passed to the dense layers.
    wi_fused_kernel_init: Optional wi_fused kernel function, passed to the
      dense layers. If None, then kernel_init will be passed instead.
    bias_init: Bias initializer.
    enable_dropout: Enables non-deterministic dropout when set to True.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    final_dropout_rate: Dropout rate used after the final layer.
    intermediate_dropout: Optional Dropout layer used after the intermediate
      layers.
    final_dropout: Optional Dropout layer used after the final layer.
    dtype: Type for the dense layer.
    out_dim: Final dimension of the output. If not set, it will be the same as
      the input dimension.
    intermediate_conv: Optional module applied to the first factor of the
      intermediate layer, after activation.
    precomputed_intermediates: whether we're using outside W_i and W_o
      computations, merely using this layer for intermediate computations.
    fuse_kernels: whether to fuse the kernels for gated activation.
    input_axis_name: Axis name for input activations.
    activations_axis_name: Axis name for intermediate activations.
    intermediate_axis_name: Axis name for output activations.
    data_sharding_constraints: Sharding constraint for data. If unspecified
      (default), sharding constraints are inferred from the data shape; see
      _get_logical_axes().
    activation_partitioning_dims: Activation partition for the intermediate
      activations.
    use_aqt: Whether to use aqt quantization.
    weight_params: Parameters for weight quantization.
    act_params: Parameters for activation quantization.
  """
  use_bias: bool
  intermediate_dim: int = 2048
  activations: Sequence[Union[str, Callable]] = ('relu',)
  kernel_init: Callable = nn.initializers.xavier_uniform()
  wi_fused_kernel_init: Optional[Callable] = None
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)
  intermediate_dropout_rate: float = 0.1
  final_dropout_rate: float = 0.1
  intermediate_dropout: Optional[nn.Module] = None
  final_dropout: Optional[nn.Module] = None
  dtype: Any = jnp.float32
  out_dim: Optional[int] = None
  intermediate_conv: Optional[nn.Module] = None
  precomputed_intermediates: bool = False
  fuse_kernels: bool = False
  input_axis_name: str = 'embed'
  activations_axis_name: str = 'mlp_activations'
  intermediate_axis_name: str = 'mlp'
  output_axis_name: str = 'embed'
  data_sharding_constraints: Optional[Tuple[str, ...]] = None
  activation_partitioning_dims: Optional[int] = 2
  use_aqt: Optional[bool] = False
  weight_params: Optional[aqt.QuantOps.WeightParams] = None
  act_params: Optional[aqt.QuantOps.ActHParams] = None
  ia3: Optional[nn.Module] = None

  @nn.compact
  def __call__(self,
               inputs,
               decode: bool = False,
               prefill: bool = False,
               prefill_lengths: Optional[Array] = None,
               *,
               enable_dropout: bool = True):
    """Applies Transformer MlpBlock module."""
    wi_fused_kernel_init = (
        self.wi_fused_kernel_init
        if self.wi_fused_kernel_init is not None else self.kernel_init)

    actual_out_dim = (
        inputs.shape[-1] if self.out_dim is None else self.out_dim)

    def dense(features, name, inputs, kernel_axis_names):
      if self.use_aqt:
        if self.weight_params is None and self.act_params is None:
          raise ValueError(
              'If use_aqt is True, either of weights or acts quantization need '
              'to be specified using arguments `weight_params` or `act_params`.'
          )
        # TODO: Push the "quantized vs not" decision down into the
        # AQT library. Currently we make that decision here, because the AQT
        # library doesn't support DenseGeneral, so there's extra reshapes here
        # whose performance impact I don't know.
        aqt_context = aqt_config.DynamicContext(
            update_bounds=False, collect_acts_stats=False)
        weight_prec = self.weight_params.prec if self.weight_params else None
        half_shift = self.weight_params.half_shift if self.weight_params else False
        aqt_hparams = aqt_flax_layers.DenseAqt.HParams(
            weight_prec=weight_prec,
            weight_half_shift=half_shift,
            quant_act=self.act_params,  # currently supports fixed bounds only.
            quant_type=aqt.QuantType.AQT,
            weight_quant_granularity=aqt_config.QuantGranularity.PER_CHANNEL,
        )
        batch, seq_len, channels = inputs.shape
        inputs = inputs.reshape((batch * seq_len, channels))
        result = aqt_flax_layers.DenseAqt(
            features=features,
            hparams=aqt_hparams,
            train=enable_dropout,
            dynamic_context=aqt_context,
            paxis_name=None,
            # No "cross-replica" reduction expressed in the XLA graph at this
            # stage. Will be imposed later, automatically, by XLA SPMD.
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            kernel_axis_names=kernel_axis_names,
            name=name,
        )(inputs, padding_mask=None)
        return result.reshape((batch, seq_len, features))
      else:
        return DenseGeneral(
            features,
            use_bias=self.use_bias,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            kernel_axis_names=kernel_axis_names,
            name=name)(
                inputs)

    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
    # TODO: don't bother w/ fusion if only a single input matrix?
    if not self.fuse_kernels:
      if self.precomputed_intermediates:
        for idx, (inpt, act_fn) in enumerate(zip(inputs, self.activations)):
          x = _convert_to_activation_function(act_fn)(inpt)
          if idx == 0 and self.intermediate_conv is not None:
            x = self.intermediate_conv(  # pylint: disable=not-callable
                x,
                decode=decode,
                prefill=prefill,
                prefill_lengths=prefill_lengths)
          activations.append(x)
      else:
        for idx, act_fn in enumerate(self.activations):
          dense_name = 'wi' if len(self.activations) == 1 else f'wi_{idx}'
          x = dense(self.intermediate_dim, dense_name, inputs,
                    (self.input_axis_name, self.intermediate_axis_name))
          x = _convert_to_activation_function(act_fn)(x)
          # N.B. We can't hook our ia3 at this step as the `intermediate_conv`
          # as that would only get applied to the gating activations in the case
          # of a gated gelu
          if idx == 0 and self.intermediate_conv is not None:
            x = self.intermediate_conv(  # pylint: disable=not-callable
                x,
                decode=decode,
                prefill=prefill,
                prefill_lengths=prefill_lengths)
          activations.append(x)
    else:
      if self.weight_params is not None or self.act_params is not None:
        # TODO: need to make quantization work with fused kernels.
        raise NotImplementedError('Quantization is not supported yet for ',
                                  'fused kernels.')
      if self.precomputed_intermediates:
        if self.out_dim is None:
          raise ValueError('Must specify mlp out_dim when using precomputed '
                           'intermediates.')
        xs = inputs
      else:
        xs = DenseGeneral(
            (len(self.activations), self.intermediate_dim),
            use_bias=self.use_bias,
            dtype=self.dtype,
            kernel_init=wi_fused_kernel_init,
            bias_init=self.bias_init,
            reshape_kernel=False,
            kernel_axis_names=(self.input_axis_name, self.activations_axis_name,
                               self.intermediate_axis_name),
            name='wi_fused')(
                inputs)
      for idx, act_fn in enumerate(self.activations):
        x = jnp.squeeze(lax.dynamic_slice_in_dim(xs, idx, 1, -2), -2)
        x = _convert_to_activation_function(act_fn)(x)
        if idx == 0 and self.intermediate_conv is not None:
          x = self.intermediate_conv(  # pylint: disable=not-callable
              x,
              decode=decode,
              prefill=prefill,
              prefill_lengths=prefill_lengths)
        activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations)

    # N.B. Actually apply ia3
    x = self.ia3(x)  # pylint: disable=not-callable

    # Apply dropout and final dense output projection.
    # TODO: Change the `None` branch to not applying dropout
    # instead of fallback to default dropout.
    if self.intermediate_dropout:
      x = self.intermediate_dropout(x, deterministic=not enable_dropout)  # pylint: disable=not-callable
    else:
      x = nn.Dropout(
          rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
              x, deterministic=not enable_dropout)  # Broadcast along length.

    # Note: We don't use `activation_partitioning.with_sharding_migration` here
    # because we do often want this 2D sharded. However, if rules are valid,
    # they should result in 2D sharding. We don't need to raise errors if both
    # result in 2D sharding (which with_sharding_migration does).
    if partitioning.get_axis_rules():
      logical_axis_resources = (
          self.data_sharding_constraints or _get_logical_axes(x))
      x = partitioning.with_sharding_constraint(
          x, logical_axis_resources=logical_axis_resources)
    else:
      x = activation_partitioning.with_sharding(
          x, self.activation_partitioning_dims)

    if self.precomputed_intermediates:
      # we fuse W_out and attention 'O' matrix outside.
      output = x
    else:
      output = dense(actual_out_dim, 'wo', x,
                     (self.intermediate_axis_name, self.output_axis_name))
      # TODO: Change the `None` branch to not applying dropout
      # instead of fallback to default dropout.
      if self.final_dropout:
        output = self.final_dropout(output, deterministic=not enable_dropout)  # pylint: disable=not-callable
      else:
        output = nn.Dropout(
            rate=self.final_dropout_rate, broadcast_dims=(-2,))(
                output, deterministic=not enable_dropout)
    return output

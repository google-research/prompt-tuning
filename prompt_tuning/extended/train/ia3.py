# Copyright 2023 Google.
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

from typing import Tuple

import flax.linen as nn
from flax.linen import partitioning
import jax.numpy as jnp
from prompt_tuning import prompts
from flaxformer.types import DType


class IA3(nn.Module):
  """IA3 scaling to use with a linear layer.

  Note:
    This module is used as the intermediate_conv module in the flaxformer
    MlpBlock. The MlpBlock only applies this intermediate conv to one of the
    parallel activation functions it uses, but because these parallel
    activations are combined with multiplication, IA3 applies an multiplicative
    scaling, and multiplication is associative we can apply the scaling to just
    that activation and get the same result as if we applied it afterwards.

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

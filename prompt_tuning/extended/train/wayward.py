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

"""An extended model for prompt tuning with a "wayward prompts" regularizer.

We have an extra term in our loss function that regularizes the learned prompt
towards the embedded representation of a discrete prompt from
Khashabi, et al. (2021) https://arxiv.org/abs/2112.08348
"""

from typing import Any, Callable, List, Mapping, Optional, Sequence

from flax import linen as nn
from flax import optim
from flax import traverse_util
from flax.core import unfreeze
import jax.numpy as jnp
import seqio
from t5x import decoding
from t5x import models

from flaxformer.types import Array


def length(x: Any) -> int:
  """A length function we can bind to in gin."""
  return len(x)


def encode_string(s: str, vocab: seqio.SentencePieceVocabulary) -> List[str]:
  """Break a string into sentence pieces."""
  return vocab.tokenizer.EncodeAsPieces(s)


def execute_initializer(init: Callable[[Array, Sequence[int]], Array],
                        rng: Array, shape: Sequence[int]) -> Array:
  """A function to execute a flax initializer outside of the .init method."""
  return init(rng, shape)


def squared_l2(x: Array) -> float:
  """Calculate the squared l2 norms of a sequence of arrays.

  Note:
    We use the squared l2 norm as things like the ranking will the same
    without needing to do the expensive sqrt.

  Args:
    x: The sequence of arrays to calculate the norm of. [T, H]

  Returns:
    The norm over the hidden dimension of the sequence of arrays. [T]
  """
  return jnp.sum(jnp.square(x), axis=1)


def squared_l2_distance(x: Array, y: Array) -> Array:
  """Calculate the squared l2 dist between arrays, normalized by length."""
  l, _ = x.shape
  return jnp.sum(squared_l2(x - y)) / l


class WaywardPromptEncoderDecoderModel(models.EncoderDecoderModel):
  """Regularize a prompt towards a discrete representation a la (Khashabi, et al., 2021)."""

  def __init__(
      self,
      module: nn.Module,
      input_vocabulary: seqio.Vocabulary,
      output_vocabulary: seqio.Vocabulary,
      optimizer_def: optim.OptimizerDef,
      decode_fn: models.DecodeFnCallable = decoding.beam_search,
      feature_converter_cls: Optional[Callable[...,
                                               seqio.FeatureConverter]] = None,
      label_smoothing: float = 0.0,
      z_loss: float = 0.0,
      loss_normalizing_factor: Optional[float] = None,
      gamma: float = 0.01,
      distance: Callable[[Array, Array], Array] = squared_l2_distance,
      discrete_prompt: Array = None,
      prompt_path: str = "encoder/prompt/prompt/prompt",
  ):
    super().__init__(
        module=module,
        input_vocabulary=input_vocabulary,
        output_vocabulary=output_vocabulary,
        optimizer_def=optimizer_def,
        decode_fn=decode_fn,
        feature_converter_cls=feature_converter_cls,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=loss_normalizing_factor,
    )
    self.gamma = gamma
    self.distance = distance
    self.discrete_prompt = discrete_prompt
    self.prompt_path = prompt_path

  def loss_fn(
      self,
      params: models.PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jnp.ndarray],
  ):
    loss, metrics = super().loss_fn(
        params=params, batch=batch, dropout_rng=dropout_rng)

    flat_params = {
        "/".join(k): v
        for k, v in traverse_util.flatten_dict(unfreeze(params)).items()
    }
    continuous_prompt = flat_params[self.prompt_path]
    wayward_loss = self.distance(continuous_prompt, self.discrete_prompt)
    return loss + self.gamma * wayward_loss, metrics

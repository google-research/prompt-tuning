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

"""Layers for adding prompts to each layer of a model."""


from typing import Callable, Optional
import flax.linen as nn
import jax.numpy as jnp
from flaxformer.architectures.t5 import t5_architecture
from flaxformer.types import Array

# pylint: disable=not-callable
# pytype: disable=not-callable


def replace_prompt(prompt: Array,
                   x_embed: Array,
                   x: Array) -> Array:
  """Replace the beginning of x_embed with prompt."""
  del x
  prompt_length = prompt.shape[1]
  return jnp.concatenate([prompt, x_embed[:, prompt_length:, :]], axis=1)


def add_prompt(prompt: Array,
               x_embed: Array,
               x: Array) -> Array:
  """Add the prompt to the beginning of x_embed."""
  del x
  prompt_length = prompt.shape[1]
  added_prompt = prompt + x_embed[:, :prompt_length, :]
  return jnp.concatenate([added_prompt, x_embed[:, prompt_length:, :]], axis=1)


class PromptEncoderLayer(t5_architecture.EncoderLayer):
  """An EncoderLayer Subclass allowing a prompt at every layer.

  This module implements something between Prompt Tuning (Lester et al. 2021)
  and Prefix Tuning (Li and Liang, 2021). When used with the `replace_prompt`
  combining function in `prompt_tuning.train.prompts`, the prompt at this layer
  of the network will overwrite the prompt segment of the current input. When
  used with the `add_prompt` combining function, it is closer to the learned
  deltas for prompt values from Qin and Eisner, 2021.

  Note:
    This approach isn't an exact match to Prefix Tuning as this layers prompt
    variable will still be projected to Q,K,V according to the learned
    projections in the frozen model while Prefix Tuning directly overwrites
    the K,V values at the beginning of the input.

    This implementation does not currently support the reparameterization of the
    prefix variable where the prefixes for each layer is sliced from a large
    block of parameters that are the result of a learned projection. Instead,
    each layer learns it own block of parameters in isolation.

    Using this module still requires using the rest of the PromptTuning
    subclasses (PromptEncoder, PromptEncoderDecoder, etc.) as those are the
    modules that handle things like updating attention masking.

  Attributes:
    prompt_factory: A factory that creates a prompt module.
  """
  prompt_factory: Optional[Callable[[], nn.Module]] = None

  def setup(self):
    super().setup()
    if self.prompt_factory is None:
      raise ValueError(f"prompt_factory is `None` on {self.__class__.__name__} "
                       ", you need a prompt factory to make prompts!")
    self.prompt = self.prompt_factory()

  def __call__(self,
               inputs: Array,
               *args,
               **kwargs):
    inputs = self.prompt(None, inputs)
    return super().__call__(inputs, *args, **kwargs)


class PromptDecoderLayer(t5_architecture.DecoderLayer):
  """A DecoderLayer Subclass allowing a prompt at every layer.

  This module implements something between Prompt Tuning (Lester et al. 2021)
  and Prefix Tuning (Li and Liang, 2021). When used with the `replace_prompt`
  combining function in `prompt_tuning.train.prompts`, the prompt at this layer
  of the network will overwrite the prompt segment of the current input. When
  used with the `add_prompt` combining function, it is closer to the learned
  deltas for prompt values from Qin and Eisner, 2021.

  Note:
    This approach isn't an exact match to Prefix Tuning as this layers prompt
    variable will still be projected to Q,K,V according to the learned
    projections in the frozen model while Prefix Tuning directly overwrites
    the K,V values at the beginning of the input.

    This implementation does not currently support the reparameterization of the
    prefix variable where the prefixes for each layer is sliced from a large
    block of parameters that are the result of a learned projection. Instead,
    each layer learns it own block of parameters in isolation.

    Using this module still requires using the rest of the PromptTuning
    subclasses (PromptEncoder, PromptEncoderDecoder, etc.) as those are the
    modules that handle things like updating attention masking.

  Attributes:
    prompt_factory: A factory that creates a prompt module.
  """
  prompt_factory: Optional[Callable[[], nn.Module]] = None

  def setup(self):
    super().setup()
    if self.prompt_factory is None:
      raise ValueError(f"prompt_factory is `None` on {self.__class__.__name__} "
                       ", you need a prompt factory to make prompts!")
    self.prompt = self.prompt_factory()

  def __call__(self,
               targets: Array,
               *args,
               **kwargs):
    # Don't add the prompt when we are in the step-by-step decode phase
    if not kwargs.get("decode", False):
      targets = self.prompt(None, targets)
    return super().__call__(targets, *args, **kwargs)

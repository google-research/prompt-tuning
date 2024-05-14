# Copyright 2024 Google.
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

"""Training subclasses of our prompt that actually do the concatenation.

The default prompt modules yield the prompts themselves which allows them to be
used, and remain consistent, in a variety of use cases. These modules are an
adapter layer between the prompt modules that generate the prompts and their
usage during training where the prompt values are concatenated to the input.
The existence of these classes give a point of customization to allow for
experimentation in how the prompt and the input are actually combined (prepend,
append, interleave, etc). This is the class that should actually be assigned to
`PROMPT` in the gin configs.
"""

from typing import Protocol
import flax.linen as nn
import jax.numpy as jnp
from prompt_tuning import prompts
from flaxformer.types import Array


class CombinationFn(Protocol):
  """Combine a prompt and the embedded input."""

  def __call__(self, prompt: Array, x_embed: Array, x: Array) -> Array:
    """Combine the `prompt` and the embedded input (`x_embed`).

    Note:
      x (the integer values of the input) is not always required but we it as
      a parameter for all combination functions so we can easily swap them out.

    Args:
      prompt: The prompt variable.
      x_embed: The embedded input.
      x: The integer tokens for the input, this is required for some
        combinations such as adding the prompt after the input.

    Returns:
      The embedded input with the prompt added to it.
    """
    pass


def prefix_prompt(prompt: Array, x_embed: Array, x: Array) -> Array:
  """Concatenate `prompt` to the beginning of `x_embed`.

  Args:
    prompt: [B, P, H] The prompt.
    x_embed: [B, T, H] The embedded input.
    x: [B, T] The non-embedded input, used for finding the lengths of examples.

  Returns:
    The input with the prompt concatenated to the front. [B, P + T, H]
  """
  del x
  return jnp.concatenate([prompt, x_embed], axis=1)


def prefix_prompt_after_bos(prompt: Array, x_embed: Array, x: Array) -> Array:
  """Concatenate `prompt` to the beginning of `x_embed`, but after BOS token.

  Args:
    prompt: [B, P, H] The prompt.
    x_embed: [B, T, H] The embedded input, with a BOS token at the front.
    x: [B, T] The non-embedded input, used for finding the lengths of examples.

  Returns:
    The input with the prompt added to the front, after the BOS token.
    [B, 1 + P + (T - 1), H]
  """
  del x
  return jnp.concatenate(
      [x_embed[:, 0, jnp.newaxis], prompt, x_embed[:, 1:]], axis=1)


def suffix_prompt(prompt: Array,
                  x_embed: Array,
                  x: Array,
                  decoder_only: bool = False,
                  before_eos: bool = True) -> Array:
  """Concatenate `prompt` to the end of inputs in `x_embed`.

  Note:
    While this function supports putting the prompt at the end of the inputs for
    a decoder only model, it is currently unclear what token value should be
    fed into the model at the first auto-regressive decoding step. The prompt
    does not have a real token associated with it so it might be best to use
    the before_eos flag resulting in the EOS being feed in. This is untested
    though.

  Args:
    prompt: [B, P, H] The prompt.
    x_embed: [B, T, H] The embedded input.
    x: [B, T] The non-embedded input, used for finding the lengths of examples.
    decoder_only: Is this a decoder only setup where the first token is a
      `0` representing BOS?
    before_eos: Should the prompt be placed before the EOS token?

  Returns:
    The input with the prompt concatenated to the back. [B, P + T, H]
  """
  b, t, h = x_embed.shape
  p = prompt.shape[1]
  # Get the length of the example (including EOS) by counting non-pad positions.
  lengths = jnp.sum(x != 0, axis=1, keepdims=True)  # [B, 1]
  # Add an extra position because BOS uses the same id (0) as padding.
  if decoder_only:
    lengths += 1
  # Move where the prompt will go back one when we are putting the EOS after it.
  if before_eos:
    lengths -= 1
  # The offsets that need to be added to example indices (0 to T + P) to index
  # into the correct batch element when flattened. So [1, 1] becomes [T + P + 1]
  batch_offsets = jnp.reshape(jnp.arange(b), (-1, 1)) * (t + p)  # [B, 1]
  # The final tensor with enough room for the input and the prompt.
  target = jnp.zeros((b * (t + p), h), dtype=x_embed.dtype)  # [B * (T + P), H]
  # Indices that enumerate the input data.
  data_indices = jnp.arange(t)  # [T]
  # Create a mask where padding (and optionally EOS) are marked with 1.
  # [T] >= [B, 1] -> [B, T]
  length_mask = data_indices >= lengths
  # Shift the index for all padding (and optionally EOS) by the prompt length
  # to make room for prompt to go right after the input.
  data_indices += (length_mask * p)  # [B, T]
  # Update the indices to index into a flattened batch
  data_indices += batch_offsets  # [B, T]
  flat_data_indices = jnp.reshape(data_indices, (-1,))  # [B * T]

  # A range that enumerates the prompt positions [P,]
  prompt_indices = jnp.arange(p)
  # Shift the positions by the initial placement to get positions in the
  # combined tensor. [B, 1] + [P,] -> [B, P]
  prompt_indices += lengths
  # Shift the positions based on which batch element they are in
  prompt_indices += batch_offsets
  flat_prompt_indices = jnp.reshape(prompt_indices, (-1,))  # [B * P]

  # Flatten the Batch and Sequence dimensions together.
  # [B, T + P, H] -> [B * (T + P), H]
  flat_x_embed = jnp.reshape(x_embed, (b * t, h))
  # [B, P, H] -> [B * P, H]
  flat_prompt = jnp.reshape(prompt, (b * p, h))

  # Set the data values, including padding, to the correct positions in the
  # flattened target, leaving room for the prompt.
  target = target.at[flat_data_indices, :].set(flat_x_embed)
  # Set the prompt values in the correct positions in the flattened array
  target = target.at[flat_prompt_indices, :].set(flat_prompt)
  # Reshape to the batched tensor we expect.
  # [B * (T + P), H] -> [B, (T + P), H]
  return jnp.reshape(target, (b, (t + p), h))


class Prompt(nn.Module):
  """Generate a Prompt and concatenate it with the input.

  This is the training time version of prompting a model. Calling the injected
  `prompt` module will generate your unbatched prompt. This model then
  replicates it for the batched input and concatenates them together.

  Attributes:
    prompt: The module that actually generates the unbatched prompt.
    combine: A function that combines the prompt and the embedded input.
  """
  prompt: nn.Module
  combine: CombinationFn = prefix_prompt

  def __call__(self, x, x_embed):
    prompt = self.prompt(x, x_embed)
    prompt = prompts.expand_to_batch(prompt, x_embed)
    # TODO: Create a minimum reproducible example and bug for the
    # pytype error when calling a function bound to the class attribute.
    #
    # Pytype is throwing a false positive here, it probably thinks
    # `self.combine` is a method call that is giving a `self` parameter but it
    # is actually just a function so there are only 2 arguments, like the type
    # annotation says.
    return self.combine(prompt, x_embed, x)  # pylint: disable=too-many-function-args

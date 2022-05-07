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

"""Create prompt-aware masks."""

from typing import Optional, Callable
import jax.numpy as jnp
from flaxformer.components.attention import dense_attention
from flaxformer.types import Array
from flaxformer.types import DType


def add_fake_prompt(
    prompt_length: int,
    multitask: bool = False,
    multitask_factorized: bool = False) -> Callable[[Array], Array]:
  """Prepend fake (active) input to update masks for prompt tuning.

  Note:
    Regardless of the location of the prompts, the beginning; interleaved; at
    the end; etc., this should always work (assuming the decoder has full
    visibility to the encoder) as the result will be `1`s that cover the prompt
    wherever it actually is.

  Args:
    prompt_length: The length of the prompt.
    multitask: True if the first token of the input is task indices (which are
      ignored in the calculation).
    multitask_factorized: True if the first two tokens of the input are language
      and task indices (which are ignored in the calculation).

  Returns:
    A callable that takes the input tokens and returns the input tokens with
    ones for the prompt prepended to them.
  """

  def _add_fake_prompt(encoder_input_tokens: Array):
    """Add ones that represent the prompt to the input tokens.

    Args:
      encoder_input_tokens: [B, L] The input tokens.

    Returns:
      The input with fake ones prepended for the prompt location.
      [B, prompt_len + L].
    """
    pad_length = prompt_length
    pad = jnp.ones((encoder_input_tokens.shape[0], pad_length),
                   dtype=encoder_input_tokens.dtype)
    if multitask:
      # If we are multitask, make sure the encoder mask doesn't include the
      # the task index which will be sliced off.
      encoder_input_tokens = encoder_input_tokens[:, 1:]
    elif multitask_factorized:
      # If we are multitask, make sure the encoder mask doesn't include the
      # the language and task indices which will be sliced off.
      encoder_input_tokens = encoder_input_tokens[:, 2:]
    return jnp.concatenate([pad, encoder_input_tokens], axis=1)

  return _add_fake_prompt


def create_prompt_encoder_mask(
    prompt_length: int) -> Callable[[Array, DType], Array]:
  """Create a prompt-aware encoder attention mask.

  Note:
    This returns a mask with an extra `prompt_length` positions on the front
    such that all input positions can attend to all prompt positions, and all
    prompt positions can attend to all input positions.

  Args:
    prompt_length: How long the prompt that will be added is.

  Returns:
    A callable that will return an attention mask when passed the inputs.
  """
  add_prompt = add_fake_prompt(prompt_length)

  def _prompt_encoder_attention_mask(encoder_input_tokens: Array,
                                     dtype: DType = jnp.float32) -> Array:
    """Create the prompt-aware encoder attention mask.

    Args:
      encoder_input_tokens: The input tokens.
      dtype: The dtype we want the mask to be.

    Returns:
      The encoder attention mask.
    """
    inputs = add_prompt(encoder_input_tokens)
    return dense_attention.make_attention_mask(inputs > 0,
                                               inputs > 0,
                                               dtype=dtype)

  return _prompt_encoder_attention_mask


def create_prompt_decoder_only_mask(
    prompt_length: int) -> Callable[[Array, Array, DType], Array]:
  """Create a prompt-aware decoder-only attention mask.

  Note:
    This mask is designed for when the prompt is applied to the decoder itself,
    it is not required for a decoder the attends to a promted encoder.

    This returns a mask with an extra `prompt_length` positions on the front
    such that all input positions can attend to all prompt positions. If
    `decoder_causal_attention` is not None, then the prompts can attend to the
    prefix of fully visible attention that is specified by the mask, otherwise
    prompts just have causal attention to earlier positions.

  Args:
    prompt_length: How long the prompt that will be added is.

  Returns:
    A callable that will return an attention mask when passed the inputs.
  """
  add_prompt = add_fake_prompt(prompt_length)

  def _prompt_decoder_only_mask(decoder_target_tokens: Array,
                                decoder_causal_attention: Optional[Array],
                                dtype: DType = jnp.float32) -> Array:
    """Create the prompt-aware decoder attention mask.

    Args:
      decoder_target_tokens: The tokens.
      decoder_causal_attention: The mask the marks which parts of the input are
        fully visible in attention.
      dtype: What dtype we want the mask to be.

    Returns:
      The decoder attention mask.
    """
    targets = add_prompt(decoder_target_tokens)
    if decoder_causal_attention is not None:
      decoder_causal_attention = add_prompt(decoder_causal_attention)
    return dense_attention.make_decoder_mask(
        decoder_target_tokens=targets,
        decoder_causal_attention=decoder_causal_attention,
        dtype=dtype,
        decoder_segment_ids=None
    )

  return _prompt_decoder_only_mask

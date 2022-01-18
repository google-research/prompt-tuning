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

"""Create prompt-aware masks with the ability to control attention between inputs and prompts."""

from typing import Set, Optional, Callable
from absl import logging
import jax.numpy as jnp
from flaxformer.components.attention import dense_attention
from flaxformer.types import Array
from flaxformer.types import DType


# Various string constants for prompt, input attention types.
INPUT_INPUT = "input->input"
INPUT_PROMPT = "input->prompt"
PROMPT_PROMPT = "prompt->prompt"
PROMPT_PROMPT_CAUSAL = "prompt->prompt;causal"
PROMPT_INPUT = "prompt->input"
AttentionTypes = (
    INPUT_INPUT,
    INPUT_PROMPT,
    PROMPT_PROMPT,
    PROMPT_PROMPT_CAUSAL,
    PROMPT_INPUT,
)


def summarize_attention(attentions: Set[str], decoder: bool = False) -> str:
  """Explain, in words, what can see what in the encoder attention."""
  attn_text = []
  for attn in AttentionTypes:
    if attn in attentions:
      q, k = attn.split("->", maxsplit=1)
      if k.endswith(";causal"):
        attn_text.append(f"\tThe {q}s can see the {k}s causally, it can only "
                         "see previous positions.")
      else:
        attn_text.append(f"\tThe {q}s can see the {k}s")
  if decoder:
    attn_text.append("\tThe targets can see the prefix and prompts, causally.")
  attn_text = "\n".join(attn_text)
  return f"Your attention configuration means that:\n{attn_text}"


def prompt_encoder_attention_mask(
    prompt_length: int,
    allowable_attention: Optional[Set[str]] = None,
    multitask: bool = False) -> Callable[[Array, DType], Array]:
  """Create the prompt-aware encoder attention mask.

  Args:
    prompt_length: How long the prompt is.
    allowable_attention: What is allowed to see what in the encoder attention.
      if missing, input->input (normal) and input->prompt attention are added.
    multitask: True if the first token of the input is the task_index which
      should not be considered as it will be removed.

  Returns:
    A callable that takes the input and returns the appropriate mask.
  """
  if allowable_attention is None:
    allowable_attention = set(AttentionTypes)
  attn_types = set(AttentionTypes)
  # Make sure we understand the attention type inputs, avoid types that cause
  # silent errors.
  if any(attn not in attn_types for attn in allowable_attention):
    raise ValueError("Attention types not understood, got "
                     f"{allowable_attention}, attention values should be from "
                     f"{AttentionTypes}")
  # Make sure input->input and input->prompt attention are used.
  if INPUT_INPUT not in allowable_attention:
    logging.warning("Not having input -> input attention doesn't make sense, "
                    "adding it to allowed attention.")
    allowable_attention.add(INPUT_INPUT)
  if INPUT_PROMPT not in allowable_attention:
    logging.warning("Not having input -> prompt attention doesn't make sense, "
                    "adding it to allowed attention.")
    allowable_attention.add(INPUT_PROMPT)
  # Prefer prompt->prompt;causal over prompt->prompt
  if (PROMPT_PROMPT in allowable_attention and
      PROMPT_PROMPT_CAUSAL in allowable_attention):
    logging.warning(
        "Having both %s and %s attention doesn't make sense, using "
        "%s because it is more specific", PROMPT_PROMPT, PROMPT_PROMPT_CAUSAL,
        PROMPT_PROMPT_CAUSAL)
    allowable_attention.discard(PROMPT_PROMPT)
  logging.info(summarize_attention(allowable_attention))

  def _prompt_encoder_attention_mask(encoder_input_tokens: Array,
                                     dtype: DType = jnp.float32) -> Array:
    """Create the prompt-aware encoder attention mask.

    Args:
      encoder_input_tokens: The input tokens.
      dtype: The data type the resulting mask should be.

    Returns:
      The encoder attention mask.
    """
    if multitask:
      encoder_input_tokens = encoder_input_tokens[:, 1:]

    full_length = prompt_length + encoder_input_tokens.shape[1]

    prompt_ones = jnp.ones((encoder_input_tokens.shape[0], prompt_length),
                           dtype=encoder_input_tokens.dtype)
    prompt_zeros = jnp.zeros((encoder_input_tokens.shape[0], prompt_length),
                             dtype=encoder_input_tokens.dtype)

    # This is a mask that creates the Input->Input full attention, accounting
    # for things like padding. In this the prompts are always visible to the
    # input but the prompts cannot see anything. This means it has Input->Input
    # and Input->Prompt attention.
    mask = dense_attention.make_attention_mask(
        jnp.concatenate([prompt_zeros, encoder_input_tokens], axis=1) > 0,
        jnp.concatenate([prompt_ones, encoder_input_tokens], axis=1) > 0,
        dtype=dtype)

    if PROMPT_INPUT in allowable_attention:
      # This is a mask where, again, the Input->Input full attention is present,
      # but it lets prompts see inputs and does NOT let inputs see prompts. The
      # lack of inputs seeing prompts will be fixed when combined with the
      # original mask.
      prompt_sees_input = dense_attention.make_attention_mask(
          jnp.concatenate([prompt_ones, encoder_input_tokens], axis=1) > 0,
          jnp.concatenate([prompt_zeros, encoder_input_tokens], axis=1) > 0,
          dtype=dtype)
      # This combines the to masks which means it includes, Input->Input
      # attention Input->Prompt attention, and Prompt->Input attention.
      mask = jnp.logical_or(mask, prompt_sees_input).astype(mask.dtype)

    # We will now add Prompt->Prompt visibility into the mask
    # This mask says all prompts can see other prompts. This covers the case of
    # Prompt->Prompt attention.
    #
    # This mask only has 1s in the prompt section of the mask, all input
    # positions will have zeros.
    within_prompt_visibility = dense_attention.make_attention_mask(
        jnp.arange(full_length) < prompt_length,
        jnp.arange(full_length) < prompt_length,
        dtype=dtype)
    if not (PROMPT_PROMPT in allowable_attention or
            PROMPT_PROMPT_CAUSAL in allowable_attention):
      # This updates the mask so a prompt can only see itself. This is needed
      # for when prompt->prompt attention isn't allowed. It results in the key
      # value being copied because it is the only one attended to. The restricts
      # the values in the Prompt-Prompt attention mask.
      #
      # This only happens when we don't have Prompt->Prompt or causal
      # Prompt->Prompt attention specified.
      within_prompt_visibility = within_prompt_visibility * jnp.eye(
          full_length, dtype=dtype)
    # For causal attention within the prompts creates a full causal mask and
    # then let the fact that the input section is all zeros remove that for us.
    elif PROMPT_PROMPT_CAUSAL in allowable_attention:
      within_prompt_visibility = (
          within_prompt_visibility *
          dense_attention.make_causal_mask(jnp.arange(full_length),
                                           dtype=dtype))
    # Update the mask with the attention allowed for the prompts.
    return mask + within_prompt_visibility

  return _prompt_encoder_attention_mask


def prompt_decoder_attention_mask(
    prompt_length: int,
    allowable_attention: Optional[Set[str]] = None,
    multitask: bool = False) -> Callable[[Array, Array, DType], Array]:
  """Create the prompt-aware decoder attention mask.

  Note:
    This is designed for when the prompt is applied to the decoder itself, the
    decoder for a prompted encoder does not require using this function. For
    that case use the pretend input and the standard encoder decoder attnetion
    mask creation code path.

  Args:
    prompt_length: How long the prompt is.
    allowable_attention: What is allowed to see what in the encoder attention.
      if missing, input->input (normal) and input->prompt attention are added.
    multitask: True if the first token of the input is the task_index which
      should not be considered as it will be removed.

  Returns:
    A callable that takes the input and returns the appropriate mask.
  """
  logging.info(summarize_attention(allowable_attention, decoder=True))

  def _prompt_decoder_attention_mask(
      decoder_target_tokens: Array,
      decoder_causal_attention: Optional[Array],
      dtype: DType = jnp.float32) -> Array:
    """Create the prompt-aware decoder attention mask.

    Note:
      This is designed for when the prompt is applied to the decoder itself, the
      decoder for a prompted encoder does not require using this function. For
      that case use the pretend input and the standard encoder decoder attention
      mask creation code path.

    Args:
      decoder_target_tokens: The tokens.
      decoder_causal_attention: The mask the marks which parts of the input are
        fully visible in attention.
      dtype: The data type the resulting mask should be.

    Returns:
      The decoder attention mask.
    """
    if multitask:
      decoder_target_tokens = decoder_target_tokens[:, 1:]
      if decoder_causal_attention is not None:
        decoder_causal_attention = decoder_causal_attention[:, 1:]
    masks = []
    pretend = jnp.ones((decoder_target_tokens.shape[0], prompt_length),
                       dtype=decoder_target_tokens.dtype)
    # This is now our targets with the extra prompt on the front. This means we
    # will have a causal mask the will include the prompts.
    pretend_targets = jnp.concatenate([pretend, decoder_target_tokens], axis=1)
    causal_mask = dense_attention.make_causal_mask(pretend_targets, dtype=dtype)

    if decoder_causal_attention is not None:
      # We create our attention mask for the causal part of our prefix-LM by
      # using the `decoder_causal_attention` as if it was the input for our
      # encoder attention make making function. This lets us reuse the code
      # for handling things like prompt->input attention. The targets are
      # always zero here so they won't be handled.
      inputs_mask = prompt_encoder_attention_mask(
          prompt_length, allowable_attention=allowable_attention)(
              decoder_causal_attention, dtype)

      # Normally we just or the input and causal mask because the input of a
      # Prefix-LM would subsume a causal mask on the input. But because we are
      # are using prompts and we allow them to not have attention we need make
      # sure the causal mask doesn't wreck out prompt attention. We create a
      # mask that is zero for the prompts and 1 everywhere else. This is
      # combined with the causal mask so the causal mask won't touch the prompt
      # positions.
      non_prompt_mask = dense_attention.make_attention_mask(
          jnp.arange(pretend_targets.shape[1]) >= prompt_length,
          # Add 1 so the 0 doesn't mask anything
          jnp.arange(pretend_targets.shape[1]) + 1,
          dtype=dtype)
      causal_mask = causal_mask * non_prompt_mask

      # Combine the decoder causal mask with the prefix mask.
      masks.append(jnp.logical_or(causal_mask, inputs_mask).astype(dtype))
    else:
      masks.append(causal_mask)

    # Padding mask. The causal mask goes into padding so we hid it here.
    masks.append(
        dense_attention.make_attention_mask(
            pretend_targets > 0, pretend_targets > 0, dtype=dtype))

    return dense_attention.combine_masks(*masks).astype(dtype)

  return _prompt_decoder_attention_mask

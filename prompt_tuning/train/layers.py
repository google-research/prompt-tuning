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

"""Flaxformer subclasses that setup prompts.

We needed to make minor modifications within several FlaxFormer method bodies.
These changes have been marked with #NEW comments.
"""

from typing import Callable, Optional
import flax.linen as nn
import jax.numpy as jnp
from flaxformer.architectures.t5 import t5_architecture
from flaxformer.types import Array

# pylint: disable=not-callable
# pytype: disable=not-callable

# We create these aliases to minimize the amount of changes required for the
# method bodies copied from flaxformer.
dense_attention = t5_architecture.dense_attention


class PromptEncoder(t5_architecture.Encoder):
  """An Encoder Subclass that adds a Prompt to embedded input."""
  prompt_factory: Optional[Callable[[], nn.Module]] = None
  add_fake_prompt_factory: Optional[Callable[[], nn.Module]] = None

  def setup(self):  # NEW
    super().setup()
    if self.prompt_factory is None:
      raise ValueError('Prompt Factory is None, you need a prompt factory to '
                       'make prompts!')
    if self.add_fake_prompt_factory is None:
      raise ValueError(
          'A add_fake_prompt_factory is required for Prompt Tuning.')
    self.prompt = self.prompt_factory()
    self.add_fake_prompt = self.add_fake_prompt_factory()

  def embed_and_combine_inputs(self,  # pytype: disable=signature-mismatch  # overriding-parameter-name-checks
                               inputs,
                               inputs_positions=None,
                               *,
                               enable_dropout: bool = True):
    """Returns the combined embedded inputs for further encoding."""
    assert inputs.ndim == 2  # (batch, len)

    if 'position_ids' in self.embedder.embedders:
      if inputs_positions is None:
        seq_length = inputs.shape[-1]
        inputs_positions = jnp.arange(seq_length)[None, :]
      embedded_inputs = self.embedder(
          token_ids=inputs, position_ids=inputs_positions)
    else:
      embedded_inputs = self.embedder(token_ids=inputs)

    # TODO: Investigate if adding the prompt works better before or
    # after `self.input_dropout` is applied. If after works better we can turn
    # this from a copy-pasted method to am override that calls the `super()`
    # version and adds the prompt to the result.
    embedded_inputs = self.prompt(inputs, embedded_inputs)  # NEW

    embedded_inputs = self.input_dropout(
        embedded_inputs, deterministic=not enable_dropout)

    # TODO: Revert this cast or move to embedder.
    embedded_inputs = embedded_inputs.astype(self.dtype)
    return embedded_inputs

  def __call__(self,  # pytype: disable=signature-mismatch  # overriding-parameter-name-checks
               inputs,
               inputs_positions=None,
               encoder_mask=None,
               *,
               enable_dropout: bool = True):
    """Applies Transformer model on the inputs.

    Args:
      inputs: The input data.
      inputs_positions: The input subsequence positions for packed examples.
      encoder_mask: The decoder self-attention mask.
      enable_dropout: Enables dropout if set to True.

    Returns:
      The output of a transformer encoder.
    """
    embedded_inputs = self.embed_and_combine_inputs(
        inputs,
        inputs_positions=inputs_positions,
        enable_dropout=enable_dropout)
    # BEGIN NEW
    logit_mask = jnp.expand_dims(
        jnp.array((self.add_fake_prompt(inputs) > 0),
                  dtype=embedded_inputs.dtype),
        axis=-1)
    # END NEW
    encoder_outputs = self.encode_from_continuous_inputs(
        embedded_inputs,
        encoder_mask=encoder_mask,
        logit_mask=logit_mask,
        enable_dropout=enable_dropout)

    if logit_mask is not None:
      encoder_outputs = logit_mask * encoder_outputs
    return encoder_outputs


class PromptDecoder(t5_architecture.Decoder):
  """A decoder class that adds a prompt to the decoder."""
  prompt_factory: Optional[Callable[[], nn.Module]] = None
  add_fake_prompt_factory: Optional[Callable[[], nn.Module]] = None

  def setup(self):  # NEW
    super().setup()
    if self.prompt_factory is None:
      raise ValueError('Prompt Factory is None, you need a prompt factory to '
                       'make prompts!')
    if self.add_fake_prompt_factory is None:
      raise ValueError(
          'A add_fake_prompt_factory is required for Prompt Tuning.')
    self.add_fake_prompt = self.add_fake_prompt_factory()
    self.prompt = self.prompt_factory()

  def embed_and_combine_inputs(  # pytype: disable=signature-mismatch  # overriding-parameter-name-checks
      self,
      decoder_input_tokens,
      decoder_positions=None,
      *,
      enable_dropout: bool = True,
      decode: bool = False,
  ):
    """Returns the combined embedded decoder inputs for further processing."""
    assert decoder_input_tokens.ndim == 2  # (batch, len)

    if 'position_ids' in self.embedder.embedders:
      if decoder_positions is None:
        seq_length = decoder_input_tokens.shape[-1]
        decoder_positions = jnp.arange(seq_length)[None, :]
      embedded_inputs = self.embedder(
          token_ids=decoder_input_tokens,
          position_ids=decoder_positions,
          decode=decode)
    else:
      embedded_inputs = self.embedder(
          token_ids=decoder_input_tokens, decode=decode)

    # BEGIN NEW
    # During training, add the prompt to the input, during prefilling, add the
    # prompt to the input, during single step decoding, don't add the prompt.
    if not decode:
      embedded_inputs = self.prompt(decoder_input_tokens, embedded_inputs)
    # END NEW

    embedded_inputs = self.input_dropout(
        embedded_inputs, deterministic=not enable_dropout)

    # TODO: Revert this cast or move to embedder.
    embedded_inputs = embedded_inputs.astype(self.dtype)
    return embedded_inputs

  def __call__(self,  # pytype: disable=signature-mismatch  # overriding-parameter-name-checks
               encoder_outputs,
               decoder_input_tokens,
               decoder_positions=None,
               decoder_mask=None,
               encoder_decoder_mask=None,
               *,
               enable_dropout: bool = True,
               decode: bool = False,
               max_decode_length: Optional[int] = None,
               prefill: bool = False,
               prefill_lengths: Optional[Array] = None):
    """Applies Transformer model on the inputs.

    TODO: For consistency it would be better to flip the order of the
    first two positional arguments here.

    Args:
      encoder_outputs: The outputs from the encoder. If None, do not attend to
        encoder outputs, resulting in a decoder only model (i.e. language
        model).
      decoder_input_tokens: The decoder input token IDs.
      decoder_positions: Decoder subsequence positions for packed examples.
      decoder_mask: Decoder self-attention mask.
      encoder_decoder_mask: The attention mask for the encoder outputs.
      enable_dropout: Enables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.
      max_decode_length: An optional integer specifying the maximum decoding
        length. Note that this is only used for defining the relative position
        embedding parameters.
      prefill: Whether to run a partial sequence to prefill the cache.
      prefill_lengths: The length of each partial sequence we are filling in the
        cache, lengths are inferred from the mask if not provided.

    Returns:
      The decoder output logits for next token prediction.
    """
    embedded_inputs = self.embed_and_combine_inputs(
        decoder_input_tokens,
        decoder_positions=decoder_positions,
        enable_dropout=enable_dropout,
        decode=decode,
    )
    # We only add the prompt when not doing step-by-step decoding, so only add
    # the fake prompt for logit mask creation when not doing decoding.
    if not decode:
      logit_mask = dense_attention.get_decoder_logit_mask(
          self.add_fake_prompt(decoder_input_tokens),  # NEW
          embedded_inputs.dtype)
    else:
      logit_mask = dense_attention.get_decoder_logit_mask(
          decoder_input_tokens,
          embedded_inputs.dtype)

    logits = self.decode_from_continuous_inputs(
        embedded_inputs,
        encoder_outputs,
        decoder_positions=decoder_positions,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        logit_mask=logit_mask,
        enable_dropout=enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length,
        prefill=prefill,
        prefill_lengths=prefill_lengths)
    return logits


class PromptEncoderDecoder(t5_architecture.EncoderDecoder):
  """An Encoder Decoder where a prompt is applied to the Encoder."""

  encoder_mask_factory: Optional[Callable[[], nn.Module]] = None
  add_fake_prompt_factory: Optional[Callable[[], nn.Module]] = None

  def setup(self):  # NEW
    super().setup()
    if self.encoder_mask_factory is None:
      raise ValueError('An encoder_mask_factory is required for Prompt Tuning.')
    if self.add_fake_prompt_factory is None:
      raise ValueError(
          'A add_fake_prompt_factory is required for Prompt Tuning.')
    self.encoder_mask = self.encoder_mask_factory()
    self.add_fake_prompt = self.add_fake_prompt_factory()

  def encode(self,
             encoder_input_tokens,
             encoder_segment_ids=None,
             encoder_positions=None,
             *,
             enable_dropout: bool = True):
    """Applies Transformer encoder-branch on the inputs.

    Args:
      encoder_input_tokens: input data to the encoder.
      encoder_segment_ids: encoder input segmentation info for packed examples.
      encoder_positions: encoder input subsequence positions for packed
        examples.
      enable_dropout: Disables dropout if set to False.

    Returns:
      encoded feature array from the transformer encoder.
    """
    # BEGIN NEW
    # Make padding attention mask.
    encoder_mask = self.encoder_mask(encoder_input_tokens, self.dtype)  # NEW
    # Add segmentation block-diagonal attention mask if using segmented data.
    if encoder_segment_ids is not None:
      raise ValueError('Prompt Tuning does not support packing.')
    # END NEW

    return self.encoder(  # pytype: disable=attribute-error
        encoder_input_tokens,
        inputs_positions=encoder_positions,
        encoder_mask=encoder_mask,
        enable_dropout=enable_dropout)

  def decode(  # pytype: disable=signature-mismatch  # overriding-parameter-name-checks
      self,
      encoded,
      encoder_input_tokens,  # only needed for masks
      decoder_input_tokens,
      decoder_target_tokens,
      encoder_segment_ids=None,
      decoder_segment_ids=None,
      decoder_positions=None,
      *,
      enable_dropout: bool = True,
      decode: bool = False,
      max_decode_length: Optional[int] = None):
    """Applies Transformer decoder-branch on encoded-input and target.

    Args:
      encoded: encoded input data from encoder.
      encoder_input_tokens: input to the encoder (only needed for masking).
      decoder_input_tokens: input token to the decoder.
      decoder_target_tokens: target token to the decoder.
      encoder_segment_ids: encoder segmentation info for packed examples.
      decoder_segment_ids: decoder segmentation info for packed examples.
      decoder_positions: decoder subsequence positions for packed examples.
      enable_dropout: Disables dropout if set to False.
      decode: Whether to prepare and use an autoregressive cache.
      max_decode_length: An optional integer specifying the maximum decoding
        length. Note that this is only used for defining the relative position
        embedding parameters.

    Returns:
      logits array from transformer decoder.
    """
    # Make the encoder input, which is used to create a mask, include tokens for
    # the prompt.
    with_fake_prompt = self.add_fake_prompt(encoder_input_tokens)  # NEW
    # Make padding attention masks.
    if decode:
      # Fast autoregressive decoding uses only a special encoder-decoder mask.
      decoder_mask = None
      encoder_decoder_mask = dense_attention.make_attention_mask(
          jnp.ones_like(decoder_target_tokens) > 0,
          with_fake_prompt > 0,  # NEW
          dtype=self.dtype)
    else:
      decoder_mask = dense_attention.make_decoder_mask(
          decoder_target_tokens=decoder_target_tokens,
          dtype=self.dtype,
          decoder_segment_ids=decoder_segment_ids)
      encoder_decoder_mask = dense_attention.make_attention_mask(
          decoder_target_tokens > 0,
          with_fake_prompt > 0,  # NEW
          dtype=self.dtype)

    # Add segmentation block-diagonal attention masks if using segmented data.
    if encoder_segment_ids is not None:
      if decode:
        raise ValueError(
            'During decoding, packing should not be used but '
            '`encoder_segment_ids` was passed to `Transformer.decode`.')
      raise ValueError('Prompt Tuning does not support packing.')  # NEW

    # When computing the logits, we don't need decoder_target_tokens, which is
    # needed for computing the loss.
    return self.decoder(
        encoded,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        enable_dropout=enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length)


class PromptDecoderOnly(t5_architecture.DecoderOnly):
  """A DecoderOnly subclass that adds prompt aware masking."""
  decoder_mask_factory: Optional[Callable[[], nn.Module]] = None

  def setup(self):
    super().setup()
    if self.decoder_mask_factory is None:
      raise ValueError('A decoder_mask_factory is required for Prompt Tuning.')
    self.decoder_mask = self.decoder_mask_factory()

  def __call__(self,
               decoder_input_tokens,
               decoder_target_tokens,
               decoder_segment_ids=None,
               decoder_positions=None,
               decoder_causal_attention=None,
               *,
               enable_dropout: bool = True,
               decode: bool = False,
               max_decode_length: Optional[int] = None,
               prefill: bool = False,
               prefill_lengths: Optional[Array] = None):
    """Applies LanguageModel on the inputs.

    This method requires both decoder_target_tokens and decoder_input_tokens,
    which is typically a shifted version of the former. For a packed dataset, it
    usually has additional processing applied. For example, the first element of
    each sequence has id 0 instead of the shifted EOS id from the previous
    sequence.

    Args:
      decoder_input_tokens: input token to the decoder.
      decoder_target_tokens: target token to the decoder.
      decoder_segment_ids: decoder segmentation info for packed examples.
      decoder_positions: decoder subsequence positions for packed examples.
      decoder_causal_attention: a binary mask indicating the "inputs" portion of
        the concatenated sequence for a prefix LM.
      enable_dropout: Disables dropout if set to False.
      decode: Whether to prepare and use an autoregressive cache.
      max_decode_length: An optional integer specifying the maximum decoding
        length. Note that this is only used for defining the relative position
        embedding parameters.
      prefill: Whether to run a partial sequence to prefill the cache.
      prefill_lengths: The length of each partial sequence we are filling in the
        cache, lengths are inferred from the mask if not provided.

    Returns:
      logits array from LanguageModel.
    """
    if decode and prefill:
      raise ValueError('Only one of `decode` and `prefill` can be set. Use '
                       '`prefill` to pre-populate the cache for Prefix LMs '
                       'before using `decode`')
    if decoder_segment_ids is not None:
      raise ValueError('Packing not supported with Prompt Tuning.')  # NEW
    if decode:
      decoder_mask = None
    else:
      decoder_mask = self.decoder_mask(  # NEW
          decoder_target_tokens=decoder_target_tokens,
          decoder_causal_attention=decoder_causal_attention,
          dtype=self.dtype)

    # We reuse Decoder class, which can optionally takes in encoded and
    # encoder_decoder_mask. These are used when Decoder is used in the context
    # of encoder-decoder model. For LM, we don't have an encoder. So set these
    # to None.
    return self.decoder(  # pytype: disable=attribute-error
        encoder_outputs=None,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=None,
        enable_dropout=enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length,
        prefill=prefill,
        prefill_lengths=prefill_lengths)

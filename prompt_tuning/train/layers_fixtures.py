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

"""Fixtures for creating runnable model chunks for tests."""

import flax.linen as nn
import jax.numpy as jnp

from prompt_tuning import masks
from prompt_tuning import prompts
from prompt_tuning.train import layers
from prompt_tuning.train import prompts as train_prompts
from flaxformer.architectures.t5 import t5_architecture
from flaxformer.architectures.t5 import t5_architecture_test_utils
from flaxformer.components import layer_norm


make_dropout = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,))
make_layer_norm = layer_norm.T5LayerNorm


def make_prompt(prompt_length: int):
  """Make a runnable Prompt."""

  def _make_prompt():
    return train_prompts.Prompt(
        prompt=prompts.Prompt(length=prompt_length))
  return _make_prompt


def make_relative_position_bias(num_attn_heads: int = 8, dtype=jnp.float32):
  """Make a runnable position bias, def'd to avoid long lambdas."""

  def _make_relative_position_bias():
    return t5_architecture_test_utils._make_relative_position_bias(  # pylint: disable=protected-access
        num_attn_heads, dtype)
  return _make_relative_position_bias


def make_encoder_attention_mask(prompt_length: int):
  """Make a runnable encoder attention mask, def'd to avoid long lambdas."""
  def _make_encoder_attention_mask():
    return masks.create_prompt_encoder_mask(prompt_length)
  return _make_encoder_attention_mask


def make_decoder_attention_mask(prompt_length):
  """Make a runnable decoder attention mask, def'd to avoid long lambdas."""

  def _make_decoder_attention_mask():
    return masks.create_prompt_decoder_only_mask(prompt_length)
  return _make_decoder_attention_mask


def make_add_fake_prompt(prompt_length: int):
  """Make a runnable add fake prompt, def'd to avoid long lambdas."""
  def _make_add_fake_prompt():
    return masks.add_fake_prompt(prompt_length)
  return _make_add_fake_prompt


def make_prompt_encoder(num_layers: int = 3,
                        num_attn_heads: int = 8,
                        prompt_length: int = 5,
                        dtype=jnp.float32):
  """Make a runnable PromptEncoder."""

  def _make_encoder_layer(shared_relative_position_bias):
    assert shared_relative_position_bias is None
    return t5_architecture.EncoderLayer(
        attention=t5_architecture_test_utils.make_attention1(
            num_attn_heads, dtype),
        mlp=t5_architecture_test_utils.make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        relative_position_bias_factory=make_relative_position_bias(
            num_attn_heads, dtype))

  def _make_prompt_encoder(*, shared_token_embedder=None):
    return layers.PromptEncoder(
        num_layers=num_layers,
        shared_token_embedder=shared_token_embedder,
        layer_factory=_make_encoder_layer,
        input_dropout_factory=make_dropout,
        output_dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        dtype=dtype,
        prompt_factory=make_prompt(prompt_length),
        add_fake_prompt_factory=make_add_fake_prompt(prompt_length)
    )

  return _make_prompt_encoder


def make_decoder(num_layers: int = 3,
                 num_attn_heads: int = 8,
                 dtype=jnp.float32):
  """Make a runnable Decoder."""

  def _make_decoder_layer(shared_relative_position_bias):
    assert shared_relative_position_bias is None
    return t5_architecture.DecoderLayer(
        self_attention=t5_architecture_test_utils.make_attention1(
            num_attn_heads, dtype),
        encoder_decoder_attention=t5_architecture_test_utils.make_attention1(
            num_attn_heads, dtype),
        mlp=t5_architecture_test_utils.make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        relative_position_bias_factory=make_relative_position_bias(
            num_attn_heads, dtype))

  def _make_decoder(*, shared_token_embedder=None):
    return t5_architecture.Decoder(
        num_layers=num_layers,
        shared_token_embedder=shared_token_embedder,
        layer_factory=_make_decoder_layer,
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        output_logits_factory=None,
        dtype=dtype
    )

  return _make_decoder


def make_prompt_decoder(num_layers: int = 3,
                        num_attn_heads: int = 8,
                        prompt_length: int = 5,
                        dtype=jnp.float32):
  """Make a runnable PromptDecoder."""

  def _make_decoder_layer(shared_relative_position_bias):
    assert shared_relative_position_bias is None
    return t5_architecture.DecoderLayer(
        self_attention=t5_architecture_test_utils.make_attention1(
            num_attn_heads, dtype),
        encoder_decoder_attention=t5_architecture_test_utils.make_attention1(
            num_attn_heads, dtype),
        mlp=t5_architecture_test_utils.make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        relative_position_bias_factory=make_relative_position_bias(
            num_attn_heads, dtype))

  def _make_decoder(*, shared_token_embedder=None):
    return layers.PromptDecoder(
        num_layers=num_layers,
        shared_token_embedder=shared_token_embedder,
        layer_factory=_make_decoder_layer,
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        output_logits_factory=None,
        token_embedder_factory=(
            lambda: t5_architecture_test_utils.make_token_emb1(71, dtype)),
        dtype=dtype,
        prompt_factory=make_prompt(prompt_length),
        add_fake_prompt_factory=make_add_fake_prompt(prompt_length))
  return _make_decoder


def make_prompt_encoder_decoder(num_layers: int = 3,
                                num_attn_heads: int = 8,
                                prompt_length: int = 5,
                                dtype=jnp.float32):
  """Make a runnable PromptEncoderDecoder."""
  return layers.PromptEncoderDecoder(
      shared_token_embedder_factory=(
          lambda: t5_architecture_test_utils.make_token_emb1(71, dtype)),
      encoder_factory=make_prompt_encoder(
          num_layers, num_attn_heads, prompt_length, dtype),
      decoder_factory=make_decoder(num_layers, num_attn_heads, dtype),
      encoder_mask_factory=make_encoder_attention_mask(prompt_length),
      add_fake_prompt_factory=make_add_fake_prompt(prompt_length))


def make_prompt_decoder_only(num_layers: int = 3,
                             num_attn_heads: int = 8,
                             prompt_length: int = 5,
                             dtype=jnp.float32):
  """Make a runnable PromptDecoderOnly."""

  return layers.PromptDecoderOnly(
      decoder_factory=make_prompt_decoder(
          num_layers, num_attn_heads, prompt_length, dtype),
      dtype=dtype,
      decoder_mask_factory=make_decoder_attention_mask(prompt_length))

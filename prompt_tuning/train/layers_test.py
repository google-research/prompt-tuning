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

"""Tests for layers.

Note:
  FlaxFormer is setup to take advantage of Gin so most of the inputs to the
  modules are factories (or more specifically, gin configurables that already
  have their arguments applied) so they expect to be called in the setup with
  very few arguments.

  We simulate this with lambdas that return Autospec'd mocks that setup to
  return instances of the class.
"""

# These lambdas are long because cause they are verbose and the mocks have
# arguments, they are not actually overly complex so disable this check.
# pylint: disable=g-long-lambda

from unittest import mock
from absl.testing import absltest
import flax.linen as nn
import jax
import jax.numpy as jnp
from prompt_tuning import masks
from prompt_tuning.train import layers
from prompt_tuning.train import layers_fixtures
from prompt_tuning.train import prompts
from flaxformer.architectures.t5 import t5_architecture
from flaxformer.architectures.t5 import t5_architecture_test_utils
from flaxformer.components import embedding
from flaxformer.components import layer_norm


class PromptEncoderTest(absltest.TestCase):

  def test_error_on_missing_prompt_factory(self):
    with self.assertRaises(ValueError):
      layers.PromptEncoder(
          layer_factory=lambda shared_relative_position_bias: mock.
          create_autospec(
              t5_architecture.EncoderLayer, spec_set=True, instance=True),
          input_dropout_factory=lambda: mock.create_autospec(
              nn.Dropout, spec_set=True, instance=True),
          output_dropout_factory=lambda: mock.create_autospec(
              nn.Dropout, spec_set=True, instance=True),
          shared_token_embedder=mock.create_autospec(
              embedding.Embed, spec_set=True, instance=True),
          layer_norm_factory=lambda: mock.create_autospec(
              layer_norm.T5LayerNorm, spec_set=True, instance=True),
          num_layers=2,
          prompt_factory=None).init(jax.random.PRNGKey(0), jnp.ones((4, 5)))

  @mock.patch.object(
      t5_architecture.embedding.MultiEmbed,
      "__call__",
      autospec=True,
      spec_set=True)
  def test_prompts_used(self, embedding_mock):
    batch_size = 4
    seq_len = 10
    embed_dim = 100
    input_tokens = jnp.ones((batch_size, seq_len))
    prompt_mock = mock.create_autospec(
        prompts.Prompt, spec_set=True, instance=True)
    embedded_input = jnp.ones((batch_size, seq_len, embed_dim))
    embedding_mock.return_value = embedded_input
    _ = layers.PromptEncoder(
        layer_factory=lambda shared_relative_position_bias: mock.
        create_autospec(
            t5_architecture.EncoderLayer, spec_set=True, instance=True),
        input_dropout_factory=lambda: mock.create_autospec(
            nn.Dropout,
            spec_set=True,
            instance=True,
            return_value=embedded_input),
        output_dropout_factory=lambda: mock.create_autospec(
            nn.Dropout, spec_set=True, instance=True),
        shared_token_embedder=mock.create_autospec(
            embedding.Embed, spec_set=True, instance=True),
        layer_norm_factory=lambda: mock.create_autospec(
            layer_norm.T5LayerNorm, spec_set=True, instance=True),
        num_layers=2,
        prompt_factory=lambda: prompt_mock,
        add_fake_prompt_factory=lambda: mock.Mock(return_value=input_tokens)
    ).init(jax.random.PRNGKey(0), input_tokens),
    # For some reason, assert_called_once_with doesn't work when using auto-spec
    self.assertEqual(prompt_mock.call_args_list[0],
                     mock.call(input_tokens, embedded_input))

  # TODO: Add checks of the computed values so we can detect
  # when a flaxformer change breaks us even if shapes don't change.
  def test_prompt_encoder_output_shape(self):
    num_layers = 3
    num_attn_heads = 8
    prompt_length = 5
    batch_size = 2
    seq_len = 10
    d_model = 13  # Hardcoded into some of the flaxformer fixtures.
    vocab_size = 71  # Hardcoded into some of the flaxformer fixtures.
    inputs = jnp.ones((batch_size, seq_len))

    make_encoder = layers_fixtures.make_prompt_encoder(
        num_layers, num_attn_heads, prompt_length)
    encoder = make_encoder(
        shared_token_embedder=t5_architecture_test_utils.make_token_emb1(
            vocab_size, jnp.float32))
    params = encoder.init(jax.random.PRNGKey(0), inputs, enable_dropout=False)

    output = encoder.apply(params, inputs, enable_dropout=False)
    self.assertEqual(
        output.shape, (batch_size, prompt_length + seq_len, d_model))


class PromptDecoderTest(absltest.TestCase):

  def test_error_on_missing_prompt_factory(self):
    input_tokens = jnp.ones((4, 5))
    with self.assertRaises(ValueError):
      layers.PromptDecoder(
          layer_factory=lambda shared_relative_position_bias: mock.
          create_autospec(
              t5_architecture.DecoderLayer, spec_set=True, instance=True),
          dropout_factory=lambda: mock.create_autospec(
              nn.Dropout, spec_set=True, instance=True),
          layer_norm_factory=lambda: mock.create_autospec(
              layer_norm.T5LayerNorm, spec_set=True, instance=True),
          num_layers=2,
          dtype=jnp.float32,
          shared_token_embedder=mock.create_autospec(
              embedding.Embed, spec_set=True, instance=True),
          prompt_factory=None,
          add_fake_prompt_factory=lambda: mock.Mock(return_value=input_tokens)
      ).init(jax.random.PRNGKey(0), jnp.ones((4, 5)))

  @mock.patch.object(
      t5_architecture.embedding.MultiEmbed,
      "__call__",
      autospec=True,
      spec_set=True)
  def test_prompt_applied(self, embedding_mock):
    batch_size = 4
    seq_len = 10
    embed_dim = 100
    input_tokens = jnp.ones((batch_size, seq_len))
    prompt_mock = mock.create_autospec(
        prompts.Prompt, spec_set=True, instance=True)
    embedded_input = jnp.ones((batch_size, seq_len, embed_dim))
    embedding_mock.return_value = embedded_input
    layers.PromptDecoder(
        layer_factory=lambda shared_relative_position_bias: mock.
        create_autospec(
            t5_architecture.DecoderLayer, spec_set=True, instance=True),
        dropout_factory=lambda: mock.create_autospec(
            nn.Dropout, spec_set=True, instance=True),
        layer_norm_factory=lambda: mock.create_autospec(
            layer_norm.T5LayerNorm, spec_set=True, instance=True),
        num_layers=2,
        dtype=jnp.float32,
        shared_token_embedder=mock.create_autospec(
            embedding.Embed, spec_set=True, instance=True),
        prompt_factory=lambda: prompt_mock,
        add_fake_prompt_factory=lambda: mock.Mock(return_value=input_tokens)
    ).apply({"params": {}},
            input_tokens,
            decode=False,
            method=layers.PromptDecoder.embed_and_combine_inputs)
    self.assertEqual(prompt_mock.call_args_list[0],
                     mock.call(input_tokens, embedded_input))

  @mock.patch.object(
      t5_architecture.embedding.MultiEmbed,
      "__call__",
      autospec=True,
      spec_set=True)
  def test_prompt_not_applied_when_decoding(self, embedding_mock):
    batch_size = 4
    seq_len = 10
    embed_dim = 100
    prompt_mock = mock.create_autospec(
        prompts.Prompt, spec_set=True, instance=True)
    embedded_input = jnp.ones((batch_size, seq_len, embed_dim))
    embedding_mock.return_value = embedded_input
    layers.PromptDecoder(
        layer_factory=lambda shared_relative_position_bias: mock.
        create_autospec(
            t5_architecture.DecoderLayer, spec_set=True, instance=True),
        dropout_factory=lambda: mock.create_autospec(
            nn.Dropout, spec_set=True, instance=True),
        layer_norm_factory=lambda: mock.create_autospec(
            layer_norm.T5LayerNorm, spec_set=True, instance=True),
        num_layers=2,
        dtype=jnp.float32,
        shared_token_embedder=mock.create_autospec(
            embedding.Embed, spec_set=True, instance=True),
        prompt_factory=lambda: prompt_mock,
        add_fake_prompt_factory=lambda: mock.Mock(return_value=jnp.ones((4, 5)))
    ).apply({"params": {}},
            jnp.ones((4, 5)),
            decode=True,
            method=layers.PromptDecoder.embed_and_combine_inputs)
    prompt_mock.assert_not_called()

  def test_decoder_shape(self):
    batch_size = 4
    seq_len = 10
    prompt_length = 5
    vocab_size = 71
    decoder = layers_fixtures.make_prompt_decoder(3, 8, prompt_length)()
    inputs = jnp.ones((batch_size, seq_len))
    params = decoder.init(
        jax.random.PRNGKey(0), None, inputs, inputs, enable_dropout=False)
    output = decoder.apply(params, None, inputs, inputs, enable_dropout=False)
    self.assertEqual(output.shape,
                     (batch_size, prompt_length + seq_len, vocab_size))


class PromptEncoderDecoderTest(absltest.TestCase):

  def test_prompt_encoder_decoder_output_shape(self):
    num_layers = 3
    num_attn_heads = 8
    prompt_length = 5
    batch_size = 2
    input_seq_len = 10
    target_seq_len = 3
    vocab_size = 71  # Hardcoded into some of the flaxformer fixtures.
    encoder_inputs = jnp.ones((batch_size, input_seq_len))
    decoder_inputs = jnp.ones((batch_size, target_seq_len))
    decoder_targets = jnp.ones((batch_size, target_seq_len))

    encoder_decoder = layers_fixtures.make_prompt_encoder_decoder(
        num_layers, num_attn_heads, prompt_length)
    params = encoder_decoder.init(jax.random.PRNGKey(0),
                                  encoder_inputs,
                                  decoder_inputs,
                                  decoder_targets,
                                  enable_dropout=False)

    output = encoder_decoder.apply(params,
                                   encoder_inputs,
                                   decoder_inputs,
                                   decoder_targets,
                                   enable_dropout=False)
    self.assertEqual(output.shape, (batch_size, target_seq_len, vocab_size))

  def test_error_on_missing_encoder_mask_factory(self):
    with self.assertRaises(ValueError):
      layers.PromptEncoderDecoder(
          encoder_factory=lambda shared_token_embedder: mock.create_autospec(
              layers.PromptEncoder, spec_set=True, instance=True),
          decoder_factory=lambda shared_token_embedder: mock.create_autospec(
              t5_architecture.Decoder, spec_set=True, instance=True),
          shared_token_embedder_factory=lambda: mock.create_autospec(
              embedding.Embed, spec_set=True, instance=True),
          encoder_mask_factory=None,
          add_fake_prompt_factory=lambda: mock.create_autospec(
              masks.add_fake_prompt, spec_set=True),
      ).init(
          jax.random.PRNGKey(0), jnp.zeros((1, 10)), jnp.zeros((1, 10)),
          jnp.zeros((1, 10)))

  def test_error_on_missing_add_fake_prompt_factory(self):
    with self.assertRaises(ValueError):
      layers.PromptEncoderDecoder(
          encoder_factory=lambda shared_token_embedder: mock.create_autospec(
              layers.PromptEncoder, spec_set=True, instance=True),
          decoder_factory=lambda shared_token_embedder: mock.create_autospec(
              t5_architecture.Decoder, spec_set=True, instance=True),
          shared_token_embedder_factory=lambda: mock.create_autospec(
              embedding.Embed, spec_set=True, instance=True),
          encoder_mask_factory=lambda: mock.create_autospec(
              masks.create_prompt_encoder_mask(10), spec_set=True),
          add_fake_prompt_factory=None,
      ).init(
          jax.random.PRNGKey(0), jnp.zeros((1, 10)), jnp.zeros((1, 10)),
          jnp.zeros((1, 10)))

  def test_error_in_encoder_on_packing(self):
    with self.assertRaises(ValueError):
      layers.PromptEncoderDecoder(
          encoder_factory=lambda shared_token_embedder: mock.create_autospec(
              layers.PromptEncoder, spec_set=True, instance=True),
          decoder_factory=lambda shared_token_embedder: mock.create_autospec(
              t5_architecture.Decoder, spec_set=True, instance=True),
          shared_token_embedder_factory=lambda: mock.create_autospec(
              embedding.Embed, spec_set=True, instance=True),
          encoder_mask_factory=lambda: mock.create_autospec(
              masks.create_prompt_encoder_mask(10), spec_set=True),
          add_fake_prompt_factory=lambda: mock.create_autospec(
              masks.add_fake_prompt, spec_set=True),
      ).apply({"params": {}},
              encoder_input_tokens=jnp.zeros((1, 10)),
              encoder_segment_ids=jnp.zeros((1, 10)),
              method=layers.PromptEncoderDecoder.encode)

  @mock.patch.object(
      layers.dense_attention,
      "make_attention_mask",
      autospec=True,
      spec_set=True)
  @mock.patch.object(
      layers.dense_attention, "make_decoder_mask", autospec=True, spec_set=True)
  def test_error_in_decode_on_packing(self, _, __):  # pylint: disable=invalid-name
    pretend_mock = mock.create_autospec(masks.add_fake_prompt, spec_set=True)
    pretend_mock.return_value = 0
    with self.assertRaises(ValueError):
      layers.PromptEncoderDecoder(
          encoder_factory=lambda shared_token_embedder: mock.create_autospec(
              layers.PromptEncoder, spec_set=True, instance=True),
          decoder_factory=lambda shared_token_embedder: mock.create_autospec(
              t5_architecture.Decoder, spec_set=True, instance=True),
          shared_token_embedder_factory=lambda: mock.create_autospec(
              embedding.Embed, spec_set=True, instance=True),
          encoder_mask_factory=lambda: mock.create_autospec(
              masks.create_prompt_encoder_mask(10), spec_set=True),
          add_fake_prompt_factory=lambda: pretend_mock,
      ).apply({"params": {}},
              encoded=jnp.zeros((1, 10, 20)),
              encoder_input_tokens=jnp.zeros((1, 10)),
              decoder_input_tokens=jnp.zeros((1, 10)),
              decoder_target_tokens=jnp.ones((1, 10)),
              encoder_segment_ids=jnp.zeros((1, 10)),
              method=layers.PromptEncoderDecoder.decode)

  # The encoder decoder mask is a multi-step process in the decoder and not
  # covered by the mask tests so test actual values here.
  def test_encoder_decoder_mask(self):
    encoder_lengths = jnp.array([3, 7, 8, 1])
    decoder_lengths = jnp.array([2, 2, 4, 1])
    prompt_length = 3
    encoder_inputs = (jnp.arange(10) < jnp.reshape(encoder_lengths,
                                                   (-1, 1))).astype(jnp.int32)
    decoder_targets = (jnp.arange(5) < jnp.reshape(decoder_lengths,
                                                   (-1, 1))).astype(jnp.int32)
    with_fake_prompt = jnp.concatenate([
        jnp.ones(
            (len(encoder_lengths), prompt_length), jnp.int32), encoder_inputs
    ],
                                       axis=1)

    pretend_mock = mock.create_autospec(masks.add_fake_prompt, spec_set=True)
    pretend_mock.return_value = with_fake_prompt
    decoder_mock = mock.create_autospec(
        t5_architecture.Decoder, spec_set=True, instance=True)

    layers.PromptEncoderDecoder(
        encoder_factory=lambda shared_token_embedder: mock.create_autospec(
            layers.PromptEncoder, spec_set=True, instance=True),
        decoder_factory=lambda shared_token_embedder: decoder_mock,
        shared_token_embedder_factory=lambda: mock.create_autospec(
            embedding.Embed, spec_set=True, instance=True),
        encoder_mask_factory=lambda: mock.create_autospec(
            masks.create_prompt_encoder_mask(10), spec_set=True),
        add_fake_prompt_factory=lambda: pretend_mock,
    ).apply({"params": {}},
            encoded=jnp.zeros((4, 10, 20)),
            encoder_input_tokens=encoder_inputs,
            decoder_input_tokens=jnp.zeros((4, 5)),
            decoder_target_tokens=decoder_targets,
            method=layers.PromptEncoderDecoder.decode)

    encoder_decoder_mask = decoder_mock.call_args_list[0][1][
        "encoder_decoder_mask"]
    for b in range(encoder_decoder_mask.shape[0]):
      encoder_len = encoder_lengths[b] + prompt_length
      decoder_len = decoder_lengths[b]
      # Dim 1 of the mask is the head dimension which is always 1 (broadcasted
      # over the heads).
      for i in range(encoder_decoder_mask.shape[2]):
        for j in range(encoder_decoder_mask.shape[3]):
          if i < decoder_len and j < encoder_len:
            self.assertEqual(encoder_decoder_mask[b, 0, i, j], 1)
          else:
            self.assertEqual(encoder_decoder_mask[b, 0, i, j], 0)

  def test_encoder_decoder_mask_decode_mode(self):
    encoder_lengths = jnp.array([3, 7, 8, 1])
    prompt_length = 3
    encoder_inputs = (jnp.arange(10) < jnp.reshape(encoder_lengths,
                                                   (-1, 1))).astype(jnp.int32)
    with_fake_prompt = jnp.concatenate([
        jnp.ones(
            (len(encoder_lengths), prompt_length), jnp.int32), encoder_inputs
    ],
                                       axis=1)

    pretend_mock = mock.create_autospec(masks.add_fake_prompt, spec_set=True)
    pretend_mock.return_value = with_fake_prompt
    decoder_mock = mock.create_autospec(
        t5_architecture.Decoder, spec_set=True, instance=True)

    layers.PromptEncoderDecoder(
        encoder_factory=lambda shared_token_embedder: mock.create_autospec(
            layers.PromptEncoder, spec_set=True, instance=True),
        decoder_factory=lambda shared_token_embedder: decoder_mock,
        shared_token_embedder_factory=lambda: mock.create_autospec(
            embedding.Embed, spec_set=True, instance=True),
        encoder_mask_factory=lambda: mock.create_autospec(
            masks.create_prompt_encoder_mask(10), spec_set=True),
        add_fake_prompt_factory=lambda: pretend_mock,
    ).apply({"params": {}},
            encoded=jnp.zeros((4, 10, 20)),
            encoder_input_tokens=encoder_inputs,
            decoder_input_tokens=jnp.zeros((4, 5)),
            decoder_target_tokens=jnp.zeros((4, 5)),
            decode=True,
            method=layers.PromptEncoderDecoder.decode)

    encoder_decoder_mask = decoder_mock.call_args_list[0][1][
        "encoder_decoder_mask"]
    for b in range(encoder_decoder_mask.shape[0]):
      encoder_len = encoder_lengths[b] + prompt_length
      # Dim 1 of the mask is the head dimension which is always 1 (broadcasted
      # over the heads).
      # In decode mode it assumes we are doing it one step at a time so
      # everything should be visible.
      for i in range(encoder_decoder_mask.shape[2]):
        for j in range(encoder_decoder_mask.shape[3]):
          if j < encoder_len:
            self.assertEqual(encoder_decoder_mask[b, 0, i, j], 1)
          else:
            self.assertEqual(encoder_decoder_mask[b, 0, i, j], 0)


class PromptDecoderOnlyTest(absltest.TestCase):

  def test_error_without_decoder_mask_factory(self):
    with self.assertRaises(ValueError):
      layers.PromptDecoderOnly(
          decoder_factory=lambda shared_token_embedder: mock.create_autospec(
              layers.PromptDecoder, spec_set=True, instance=True),
          dtype=jnp.float32,
          decoder_mask_factory=None,
      ).init(jax.random.PRNGKey(0), jnp.ones((4, 5)), jnp.ones((4, 5)))

  def test_error_when_segment_ids_are_provided(self):
    with self.assertRaises(ValueError):
      layers.PromptDecoderOnly(
          decoder_factory=lambda shared_token_embedder: mock.create_autospec(
              layers.PromptDecoder, spec_set=True, instance=True),
          dtype=jnp.float32,
          decoder_mask_factory=lambda: mock.create_autospec(
              masks.create_prompt_decoder_only_mask(10), spec_set=True),
      ).init(
          jax.random.PRNGKey(0),
          jnp.ones((4, 5)),
          jnp.ones((4, 5)),
          decoder_segment_ids=jnp.ones((4, 5)))

  def test_no_mask_on_decode(self):
    decoder_mask_mock = mock.create_autospec(
        masks.create_prompt_decoder_only_mask(10), spec_set=True)
    layers.PromptDecoderOnly(
        decoder_factory=lambda shared_token_embedder: mock.create_autospec(
            layers.PromptDecoder, spec_set=True, instance=True),
        dtype=jnp.float32,
        decoder_mask_factory=lambda: decoder_mask_mock,
    ).init(
        jax.random.PRNGKey(0), jnp.ones((4, 5)), jnp.ones((4, 5)), decode=True)
    decoder_mask_mock.assert_not_called()

  def test_decoder_mask_on_prefill(self):
    # We can't really autospec the closure that the decoder mask factory returns
    # so use a normal mock.
    decoder_mask_mock = mock.Mock()
    target_tokens = jnp.ones((4, 5))
    decoder_causal = jnp.ones((4, 5))
    layers.PromptDecoderOnly(
        decoder_factory=lambda shared_token_embedder: mock.create_autospec(
            layers.PromptDecoder, spec_set=True, instance=True),
        dtype=jnp.float32,
        decoder_mask_factory=lambda: decoder_mask_mock,
    ).init(
        jax.random.PRNGKey(0),
        jnp.ones((4, 5)),
        target_tokens,
        decode=False,
        prefill=True,
        decoder_causal_attention=decoder_causal)
    self.assertEqual(
        decoder_mask_mock.call_args_list[0],
        mock.call(
            decoder_target_tokens=target_tokens,
            decoder_causal_attention=decoder_causal,
            dtype=jnp.float32))

  def test_decoder_mask_on_train(self):
    # We can't really autospec the closure that the decoder mask factory returns
    # so use a normal mock.
    decoder_mask_mock = mock.Mock()
    target_tokens = jnp.ones((4, 5))
    decoder_causal = jnp.ones((4, 5))
    layers.PromptDecoderOnly(
        decoder_factory=lambda shared_token_embedder: mock.create_autospec(
            layers.PromptDecoder, spec_set=True, instance=True),
        dtype=jnp.float32,
        decoder_mask_factory=lambda: decoder_mask_mock,
    ).init(
        jax.random.PRNGKey(0),
        jnp.ones((4, 5)),
        target_tokens,
        decode=False,
        prefill=False,
        decoder_causal_attention=decoder_causal)
    self.assertEqual(
        decoder_mask_mock.call_args_list[0],
        mock.call(
            decoder_target_tokens=target_tokens,
            decoder_causal_attention=decoder_causal,
            dtype=jnp.float32))

  def test_decoder_only_shape(self):
    num_layers = 3
    num_attn_heads = 8
    prompt_length = 5
    batch_size = 2
    seq_len = 10
    vocab_size = 71  # Hardcoded into some of the flaxformer fixtures.
    decoder_inputs = jnp.ones((batch_size, seq_len))
    decoder_targets = jnp.ones((batch_size, seq_len))

    decoder_only = layers_fixtures.make_prompt_decoder_only(
        num_layers, num_attn_heads, prompt_length)
    params = decoder_only.init(jax.random.PRNGKey(0),
                               decoder_inputs,
                               decoder_targets,
                               enable_dropout=False)

    output = decoder_only.apply(params,
                                decoder_inputs,
                                decoder_targets,
                                enable_dropout=False)
    self.assertEqual(output.shape,
                     (batch_size, prompt_length + seq_len, vocab_size))


if __name__ == "__main__":
  absltest.main()

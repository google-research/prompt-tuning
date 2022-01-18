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

"""Tests for prompt_tuning.masks."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from prompt_tuning import masks


def create_input(lengths):
  input_tokens = jnp.ones((len(lengths), jnp.max(lengths)), dtype=jnp.float32)
  input_tokens = input_tokens * (
      jnp.arange(jnp.max(lengths)) < jnp.reshape(lengths, (-1, 1)))
  return input_tokens, lengths


def create_decoder_input(prefix_lengths, target_lengths):
  targets, example_lengths = create_input(prefix_lengths + target_lengths)
  causal, _ = create_input(prefix_lengths)
  causal = jnp.concatenate([
      causal,
      jnp.zeros((causal.shape[0], jnp.max(example_lengths) - causal.shape[1]))
  ],
                           axis=1)
  return targets, causal, example_lengths


class CreatePromptDecoderOnlyMaskTest(parameterized.TestCase):

  @parameterized.parameters((jnp.float32,), (jnp.bfloat16,))
  def test_dtype(self, dtype):
    prompt_length = 5
    targets, causal, _ = create_decoder_input(
        jnp.array([2, 3, 4]), jnp.array([5, 6, 8]))
    mask_fn = masks.create_prompt_decoder_only_mask(prompt_length)
    mask_fn = functools.partial(mask_fn, dtype=dtype)
    mask = jax.jit(mask_fn)(targets, causal)

    self.assertEqual(mask.dtype, dtype)

  @parameterized.parameters((4,), (10,), (100,))
  def test_mask_shape(self, prompt_length):
    targets, causal, _ = create_decoder_input(
        jnp.array([2, 3, 4]), jnp.array([5, 6, 8]))
    mask_fn = masks.create_prompt_decoder_only_mask(prompt_length)
    mask = jax.jit(mask_fn)(targets, causal)
    self.assertEqual(mask.shape,
                     (targets.shape[0],
                      1,
                      targets.shape[1] + prompt_length,
                      targets.shape[1] + prompt_length))

  def test_full_causal_mask(self):
    prompt_length = 5
    targets, _, example_lengths = create_decoder_input(
        jnp.array([2, 3, 4]), jnp.array([5, 6, 8]))
    mask_fn = masks.create_prompt_decoder_only_mask(prompt_length)
    mask = jax.jit(mask_fn)(targets, None)

    batch, heads, query, key = mask.shape
    lengths = prompt_length + example_lengths
    for b in range(batch):
      for h in range(heads):
        for q in range(query):
          for k in range(key):
            if q < k:
              self.assertEqual(mask[b, h, q, k].item(), 0)
            if q >= k and q < lengths[b]:
              self.assertEqual(mask[b, h, q, k].item(), 1)

  def test_prefix_visible_mask(self):
    prompt_length = 5
    prefix_lengths = jnp.array([2, 3, 4])
    targets, causal, example_lengths = create_decoder_input(
        prefix_lengths, jnp.array([5, 6, 8]))
    mask_fn = masks.create_prompt_decoder_only_mask(prompt_length)
    mask = jax.jit(mask_fn)(targets, causal)

    batch, heads, query, key = mask.shape
    lengths = prompt_length + example_lengths
    prefix_lengths = prefix_lengths + prompt_length
    for b in range(batch):
      for h in range(heads):
        for q in range(query):
          for k in range(key):
            if q < prefix_lengths[b] and k < prefix_lengths[b]:
              self.assertEqual(mask[b, h, q, k].item(), 1)
            if q >= prefix_lengths[b]:
              if q < k:
                self.assertEqual(mask[b, h, q, k].item(), 0)
              if q >= k and q < lengths[b]:
                self.assertEqual(mask[b, h, q, k].item(), 1)


class CreatePromptEncoderAttentionMaskTest(parameterized.TestCase):

  @parameterized.parameters((jnp.float32,), (jnp.bfloat16,))
  def test_dtype(self, dtype):
    prompt_length = 5
    inputs, _ = create_input(jnp.array([5, 6, 8]))
    mask_fn = masks.create_prompt_encoder_mask(prompt_length)
    mask_fn = functools.partial(mask_fn, dtype=dtype)
    mask = jax.jit(mask_fn)(inputs)
    self.assertEqual(mask.dtype, dtype)

  @parameterized.parameters((4,), (10,), (100,))
  def test_mask_shape(self, prompt_length):
    inputs, _ = create_input(jnp.array([5, 6, 8]))
    mask_fn = masks.create_prompt_encoder_mask(prompt_length)
    mask = jax.jit(mask_fn)(inputs)
    self.assertEqual(mask.shape,
                     (inputs.shape[0],
                      1,
                      inputs.shape[1] + prompt_length,
                      inputs.shape[1] + prompt_length))

  def test_mask_prompt_visible(self):
    prompt_length = 5
    inputs, example_lengths = create_input(jnp.array([5, 6, 8]))
    mask_fn = masks.create_prompt_encoder_mask(prompt_length)
    mask = jax.jit(mask_fn)(inputs)

    batch, heads, query, key = mask.shape
    lengths = prompt_length + example_lengths
    for b in range(batch):
      for h in range(heads):
        for q in range(query):
          for k in range(key):
            if q < lengths[b] and k < lengths[b]:
              self.assertEqual(mask[b, h, q, k].item(), 1)
            else:
              self.assertEqual(mask[b, h, q, k].item(), 0)


class AddFakePromptTest(parameterized.TestCase):
  """Tests that we update prompted encoder inputs for a non-prompted decoder.

  This function can also be used for a prompted encoder that always allows
  fully visible attention in the prompt.
  """

  @parameterized.named_parameters([
      dict(testcase_name="multitask", multitask=True),
      dict(testcase_name="single_task", multitask=False)
  ])
  def test_add_fake_prompt(self, multitask):
    encoder_input_tokens, _ = create_input(jnp.array([5, 6, 7, 6, 1]))
    prompt_length = 4

    pretend_fn = masks.add_fake_prompt(prompt_length, multitask)
    pretend = jax.jit(pretend_fn)(encoder_input_tokens)

    length = encoder_input_tokens.shape[1] + prompt_length
    if multitask:
      length = length - 1
    self.assertEqual(pretend.shape, (encoder_input_tokens.shape[0], length))
    self.assertTrue(np.all(pretend[:, :prompt_length] == 1))
    if multitask:
      np.testing.assert_allclose(pretend[:, prompt_length:],
                                 encoder_input_tokens[:, 1:])
    else:
      np.testing.assert_allclose(pretend[:, prompt_length:],
                                 encoder_input_tokens)


if __name__ == "__main__":
  absltest.main()

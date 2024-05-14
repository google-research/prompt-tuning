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

"""Tests for prompt_tuning.advanced.masks."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from prompt_tuning.extended import masks


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


class PromptDecoderAttentionTest(parameterized.TestCase):
  """This tests the creation of prompt masks for decoder.

  Note:
    This function is used when the prompt is applied to the decoder, it is not
    needed for decoder that acts on a prompted encoder.
  """

  @parameterized.named_parameters([
      dict(testcase_name="multitask", multitask=True),
      dict(testcase_name="single_task", multitask=False)
  ])
  def test_prompt_decoder_attention_causal_targets(self, multitask):
    prefix_lengths = jnp.array([4, 6, 2, 3, 5])
    target_lengths = jnp.array([2, 1, 3, 1, 1])
    prompt_length = 4
    targets, causal, example_lengths = create_decoder_input(
        prefix_lengths, target_lengths)

    mask_fn = masks.prompt_decoder_attention_mask(
        prompt_length,
        set(),
        multitask=multitask,
    )
    mask = jax.jit(mask_fn)(targets, causal)

    if multitask:
      example_lengths = example_lengths - 1

    for b in range(mask.shape[0]):
      length = example_lengths[b] + prompt_length
      # Start at the end of the prefix
      for q in range(prefix_lengths[b] + prompt_length, mask.shape[2]):
        # Everything in the past (outside of padding) should be visible to the
        # target decoder.
        for k in range(mask.shape[3]):
          if q >= length or k >= length:
            self.assertEqual(mask[b, 1, q, k], 0)
          else:
            # Make sure we can only see causally.
            if q < k:
              self.assertEqual(mask[b, 1, q, k], 0)
            else:
              self.assertEqual(mask[b, 1, q, k], 1)

  @parameterized.named_parameters([
      dict(testcase_name="multitask", multitask=True),
      dict(testcase_name="single_task", multitask=False)
  ])
  def test_prompt_decoder_attention_causal_inputs(self, multitask):
    prefix_lengths = jnp.array([4, 6, 2, 3, 5])
    target_lengths = jnp.array([2, 1, 3, 1, 1])
    prompt_length = 4
    targets, causal, example_lengths = create_decoder_input(
        prefix_lengths, target_lengths)

    # Zero out the causal attention mask to the whole thing is causal.
    causal = jnp.zeros_like(causal)

    mask_fn = masks.prompt_decoder_attention_mask(
        prompt_length, set(), multitask=multitask)
    mask = jax.jit(mask_fn)(targets, causal)

    if multitask:
      example_lengths = example_lengths - 1

    for b in range(mask.shape[0]):
      example_length = example_lengths[b] + prompt_length
      for q in range(prompt_length, example_length):
        for k in range(mask.shape[3]):
          if q >= example_length or k >= example_length:
            self.assertEqual(mask[b, 1, q, k], 0)
          else:
            if k <= q:
              self.assertEqual(mask[b, 1, q, k], 1)
            else:
              self.assertEqual(mask[b, 1, q, k], 0)

  @parameterized.named_parameters([
      dict(testcase_name="multitask", multitask=True),
      dict(testcase_name="single_task", multitask=False)
  ])
  def test_prompt_decoder_attention_causal_inputs_with_none(self, multitask):
    prefix_lengths = jnp.array([4, 6, 2, 3, 5])
    target_lengths = jnp.array([2, 1, 3, 1, 1])
    prompt_length = 4
    targets, causal, example_lengths = create_decoder_input(
        prefix_lengths, target_lengths)

    causal = None

    mask_fn = masks.prompt_decoder_attention_mask(
        prompt_length, set(), multitask=multitask)
    mask = jax.jit(mask_fn)(targets, causal)

    if multitask:
      example_lengths = example_lengths - 1

    for b in range(mask.shape[0]):
      example_length = example_lengths[b] + prompt_length
      for q in range(prompt_length, example_length):
        for k in range(mask.shape[3]):
          if q >= example_length or k >= example_length:
            self.assertEqual(mask[b, 1, q, k], 0)
          else:
            if k <= q:
              self.assertEqual(mask[b, 1, q, k], 1)
            else:
              self.assertEqual(mask[b, 1, q, k], 0)

  @parameterized.named_parameters([
      dict(testcase_name="multitask", multitask=True),
      dict(testcase_name="single_task", multitask=False)
  ])
  def test_prompt_decoder_attention_full_causal(self, multitask):
    """This test is a mixture of other tests but it is a critical setting."""
    prefix_lengths = jnp.array([10, 3, 8, 2, 9, 10])
    target_lengths = jnp.array([3, 2, 6, 1, 1, 2])
    prompt_length = 10
    targets, causal, example_lengths = create_decoder_input(
        prefix_lengths, target_lengths)

    # Zero out the causal attention mask to the whole thing is causal.
    causal = jnp.zeros_like(causal)

    mask_fn = masks.prompt_decoder_attention_mask(
        prompt_length,
        set(("prompt->prompt;causal",)),
        multitask=multitask)
    mask = jax.jit(mask_fn)(targets, causal)

    if multitask:
      example_lengths = example_lengths - 1

    for b in range(mask.shape[0]):
      example_length = example_lengths[b] + prompt_length
      for q in range(mask.shape[2]):
        for k in range(mask.shape[3]):
          if q >= example_length or k >= example_length:
            self.assertEqual(mask[b, 1, q, k], 0)
          elif k <= q:
            self.assertEqual(mask[b, 1, q, k], 1)
          else:
            self.assertEqual(mask[b, 1, q, k], 0)

  @parameterized.named_parameters([
      dict(testcase_name="multitask", multitask=True),
      dict(testcase_name="single_task", multitask=False)
  ])
  def test_prompt_decoder_attention_full_causal_with_none(self, multitask):
    """This test is a mixture of other tests but it is a critical setting."""
    prefix_lengths = jnp.array([10, 3, 8, 2, 9, 10])
    target_lengths = jnp.array([3, 2, 6, 1, 1, 2])
    prompt_length = 10
    targets, causal, example_lengths = create_decoder_input(
        prefix_lengths, target_lengths)

    # Zero out the causal attention mask to the whole thing is causal.
    causal = None

    mask_fn = masks.prompt_decoder_attention_mask(
        prompt_length,
        set(("prompt->prompt;causal",)),
        multitask=multitask)
    mask = jax.jit(mask_fn)(targets, causal)

    if multitask:
      example_lengths = example_lengths - 1

    for b in range(mask.shape[0]):
      example_length = example_lengths[b] + prompt_length
      for q in range(mask.shape[2]):
        for k in range(mask.shape[3]):
          if q >= example_length or k >= example_length:
            self.assertEqual(mask[b, 1, q, k], 0)
          elif k <= q:
            self.assertEqual(mask[b, 1, q, k], 1)
          else:
            self.assertEqual(mask[b, 1, q, k], 0)

  @parameterized.named_parameters([
      dict(testcase_name="multitask", multitask=True),
      dict(testcase_name="single_task", multitask=False)
  ])
  def test_prompt_decoder_attention_input_visibility(self, multitask):
    prefix_lengths = jnp.array([4, 6, 2, 3, 5])
    target_lengths = jnp.array([2, 1, 3, 1, 1])
    prompt_length = 4
    targets, causal, example_lengths = create_decoder_input(
        prefix_lengths, target_lengths)

    mask_fn = masks.prompt_decoder_attention_mask(
        prompt_length,
        set(),
        multitask=multitask,
    )
    mask = jax.jit(mask_fn)(targets, causal)

    if multitask:
      example_lengths = example_lengths - 1
      prefix_lengths = prefix_lengths - 1

    for b in range(mask.shape[0]):
      example_length = example_lengths[b] + prompt_length
      for q in range(prompt_length, example_length):
        for k in range(mask.shape[3]):
          if q >= example_length or k >= example_length:
            self.assertEqual(mask[b, 1, q, k], 0, f"[{b}, {q}, {k}]")
          else:
            # We should see anything in the input or prompt
            if k < (prefix_lengths[b] + prompt_length):
              self.assertEqual(mask[b, 1, q, k], 1, f"[{b}, {q}, {k}]")
            elif k <= q:
              self.assertEqual(mask[b, 1, q, k], 1, f"[{b}, {q}, {k}]")
            else:
              self.assertEqual(mask[b, 1, q, k], 0, f"[{b}, {q}, {k}]")

  @parameterized.named_parameters([
      dict(testcase_name="multitask", multitask=True),
      dict(testcase_name="single_task", multitask=False)
  ])
  def test_prompt_decoder_attention_prompt_prompt(self, multitask):
    prefix_lengths = jnp.array([4, 6, 2, 3, 5])
    target_lengths = jnp.array([2, 1, 3, 1, 1])
    prompt_length = 4
    targets, causal, example_lengths = create_decoder_input(
        prefix_lengths, target_lengths)

    mask_fn = masks.prompt_decoder_attention_mask(
        prompt_length,
        {"prompt->prompt"},
        multitask=multitask,
    )
    mask = jax.jit(mask_fn)(targets, causal)

    if multitask:
      example_lengths = example_lengths - 1

    for b in range(mask.shape[0]):
      for q in range(prompt_length):
        for k in range(prompt_length):
          self.assertEqual(mask[b, 1, q, k], 1)

  @parameterized.named_parameters([
      dict(testcase_name="multitask", multitask=True),
      dict(testcase_name="single_task", multitask=False)
  ])
  def test_prompt_decoder_attention_prompt_input_visible(self, multitask):
    prefix_lengths = jnp.array([4, 6, 2, 3, 5])
    target_lengths = jnp.array([2, 1, 3, 1, 1])
    prompt_length = 4
    targets, causal, _ = create_decoder_input(prefix_lengths, target_lengths)

    mask_fn = masks.prompt_decoder_attention_mask(
        prompt_length,
        {"prompt->input"},
        multitask=multitask,
    )
    mask = jax.jit(mask_fn)(targets, causal)

    if multitask:
      prefix_lengths = prefix_lengths - 1

    for b in range(mask.shape[0]):
      for q in range(prompt_length):
        for k in range(prompt_length, prompt_length + prefix_lengths[b]):
          self.assertEqual(mask[b, 1, q, k], 1)

  @parameterized.named_parameters([
      dict(testcase_name="multitask", multitask=True),
      dict(testcase_name="single_task", multitask=False)
  ])
  def test_prompt_decoder_attention_prompt_input_not_visible(self, multitask):
    prefix_lengths = jnp.array([4, 6, 2, 3, 5])
    target_lengths = jnp.array([2, 1, 3, 1, 1])
    prompt_length = 4
    targets, causal, example_lengths = create_decoder_input(
        prefix_lengths, target_lengths)

    mask_fn = masks.prompt_decoder_attention_mask(
        prompt_length,
        set(),
        multitask=multitask,
    )
    mask = jax.jit(mask_fn)(targets, causal)

    if multitask:
      example_lengths = example_lengths - 1

    for b in range(mask.shape[0]):
      for q in range(prompt_length):
        for k in range(prompt_length, prompt_length + prefix_lengths[b]):
          self.assertEqual(mask[b, 1, q, k], 0)

  @parameterized.named_parameters([
      dict(testcase_name="multitask", multitask=True),
      dict(testcase_name="single_task", multitask=False)
  ])
  def test_prompt_decoder_attention_prompt_prompt_self_only(self, multitask):
    prefix_lengths = jnp.array([4, 6, 2, 3, 5])
    target_lengths = jnp.array([2, 1, 3, 1, 1])
    prompt_length = 4
    targets, causal, example_lengths = create_decoder_input(
        prefix_lengths, target_lengths)

    mask_fn = masks.prompt_decoder_attention_mask(
        prompt_length,
        set(),
        multitask=multitask,
    )
    mask = jax.jit(mask_fn)(targets, causal)

    if multitask:
      example_lengths = example_lengths - 1

    for b in range(mask.shape[0]):
      for q in range(prompt_length):
        for k in range(prompt_length):
          if q == k:
            self.assertEqual(mask[b, 1, q, k], 1)
          else:
            self.assertEqual(mask[b, 1, q, k], 0)

  @parameterized.named_parameters([
      dict(testcase_name="multitask", multitask=True),
      dict(testcase_name="single_task", multitask=False)
  ])
  def test_prompt_decoder_attention_prefix_visibility_causal(self, multitask):
    prefix_lengths = jnp.array([4, 6, 2, 3, 5])
    target_lengths = jnp.array([2, 1, 3, 1, 1])
    prompt_length = 4
    targets, causal, example_lengths = create_decoder_input(
        prefix_lengths, target_lengths)

    mask_fn = masks.prompt_decoder_attention_mask(
        prompt_length,
        {"prompt->prompt;causal"},
        multitask=multitask,
    )
    mask = jax.jit(mask_fn)(targets, causal)

    if multitask:
      example_lengths = example_lengths - 1

    for b in range(mask.shape[0]):
      for q in range(prompt_length):
        for k in range(prompt_length):
          if q >= k:  # k is in the past and visible.
            self.assertEqual(mask[b, 1, q, k], 1)
          else:  # k is in the future and should be hidden.
            self.assertEqual(mask[b, 1, q, k], 0)

  @parameterized.product(
      dtype=[jnp.bfloat16, jnp.float32],
      attentions=[{"prompt->input"}, {"prompt->prompt"},
                  set()],
      multitask=[True, False])
  def test_mask_dtype(self, dtype, attentions, multitask):
    prefix_lengths = jnp.array([4, 6, 2, 3, 5])
    target_lengths = jnp.array([2, 1, 3, 1, 1])
    prompt_length = 4
    targets, causal, _ = create_decoder_input(prefix_lengths, target_lengths)

    mask_fn = masks.prompt_decoder_attention_mask(
        prompt_length,
        attentions,
        multitask=multitask,
    )
    mask_fn = functools.partial(mask_fn, dtype=dtype)
    mask = jax.jit(mask_fn)(targets, causal)

    self.assertEqual(mask.dtype, dtype)

  @parameterized.named_parameters([
      dict(testcase_name="multitask", multitask=True),
      dict(testcase_name="single_task", multitask=False)
  ])
  def test_mask_shape(self, multitask):
    prefix_lengths = jnp.array([4, 6, 2, 3, 5])
    target_lengths = jnp.array([2, 1, 3, 1, 1])
    prompt_length = 4
    targets, causal, example_lengths = create_decoder_input(
        prefix_lengths, target_lengths)

    mask_fn = masks.prompt_decoder_attention_mask(
        prompt_length,
        set(),
        multitask=multitask,
    )
    mask = jax.jit(mask_fn)(targets, causal)

    length = jnp.max(example_lengths) + prompt_length
    if multitask:
      length = length - 1
    self.assertEqual(mask.shape, (targets.shape[0], 1, length, length))


class PromptEncoderAttentionTest(parameterized.TestCase):
  """Tests for masking of a prompted encoder."""

  @parameterized.named_parameters([
      dict(testcase_name="multitask", multitask=True),
      dict(testcase_name="single_task", multitask=False)
  ])
  def test_prompt_input_attention(self, multitask):
    encoder_input_tokens, lengths = create_input(jnp.array([5, 6, 7, 6, 1]))
    prompt_length = 4

    mask_fn = masks.prompt_encoder_attention_mask(
        prompt_length,
        {"prompt->input", "prompt->prompt"},
        multitask=multitask,
    )
    mask = jax.jit(mask_fn)(encoder_input_tokens)

    if multitask:
      lengths = lengths - 1

    for b in range(mask.shape[0]):
      length = lengths[b] + prompt_length
      for q in range(mask.shape[2]):
        for k in range(mask.shape[3]):
          # If either location is in padding, mask should be zero
          if q >= length or k >= length:
            self.assertEqual(mask[b, 1, q, k], 0)
          # Everything else is allowed to see each other
          else:
            self.assertEqual(mask[b, 1, q, k], 1)

  @parameterized.named_parameters([
      dict(testcase_name="multitask", multitask=True),
      dict(testcase_name="single_task", multitask=False)
  ])
  def test_prompt_self_only_mask(self, multitask):
    encoder_input_tokens, lengths = create_input(jnp.array([1, 2, 6, 3, 2, 5]))
    prompt_length = 10

    mask_fn = masks.prompt_encoder_attention_mask(
        prompt_length,
        set(),
        multitask=multitask,
    )
    mask = jax.jit(mask_fn)(encoder_input_tokens)

    if multitask:
      lengths = lengths - 1

    for b in range(mask.shape[0]):
      length = lengths[b] + prompt_length
      for q in range(mask.shape[2]):
        for k in range(mask.shape[3]):
          # If either location is in the padding, mask should be zero.
          if q >= length or k >= length:
            self.assertEqual(mask[b, 1, q, k], 0)
          # If q is in the prompt
          elif q < prompt_length:
            # It should be able to see itself,
            if k == q:
              self.assertEqual(mask[b, 1, q, k], 1)
            # But nothing else.
            else:
              self.assertEqual(mask[b, 1, q, k], 0)
          # If q is in the inputs it should see k if it is an input or a prompt
          # (we cover the case of the invalid k above).
          elif q >= prompt_length:
            self.assertEqual(mask[b, 1, q, k], 1)

  @parameterized.named_parameters([
      dict(testcase_name="multitask", multitask=True),
      dict(testcase_name="single_task", multitask=False)
  ])
  def test_prompt_prompt_attention(self, multitask):
    encoder_input_tokens, lengths = create_input(
        jnp.array([10, 5, 2, 1, 2, 5, 10]))
    prompt_length = 5

    mask_fn = masks.prompt_encoder_attention_mask(
        prompt_length,
        {"prompt->prompt"},
        multitask=multitask,
    )
    mask = jax.jit(mask_fn)(encoder_input_tokens)

    if multitask:
      lengths = lengths - 1

    for b in range(mask.shape[0]):
      length = lengths[b] + prompt_length
      for q in range(mask.shape[2]):
        for k in range(mask.shape[3]):
          # If either location is in the padding, mask should be zero.
          if q >= length or k >= length:
            self.assertEqual(mask[b, 1, q, k], 0)
          # If q is in the prompt
          elif q < prompt_length:
            # it should be able to see other prompts,
            if k < prompt_length:
              self.assertEqual(mask[b, 1, q, k], 1)
            # but not able to see the inputs
            else:
              self.assertEqual(mask[b, 1, q, k], 0)
          # If q is in the inputs it should see k if it is an input or a prompt
          # (we cover the case of the invalid k above).
          elif q >= prompt_length:
            self.assertEqual(mask[b, 1, q, k], 1)

  @parameterized.named_parameters([
      dict(testcase_name="multitask", multitask=True),
      dict(testcase_name="single_task", multitask=False)
  ])
  def test_prompt_prompt_causal_attention(self, multitask):
    encoder_input_tokens, lengths = create_input(jnp.array([8, 1, 3, 4, 1, 5]))
    prompt_length = 4

    mask_fn = masks.prompt_encoder_attention_mask(
        prompt_length,
        {"prompt->prompt;causal"},
        multitask=multitask,
    )
    mask = jax.jit(mask_fn)(encoder_input_tokens)

    if multitask:
      lengths = lengths - 1

    for b in range(mask.shape[0]):
      length = lengths[b] + prompt_length
      for q in range(mask.shape[2]):
        for k in range(mask.shape[3]):
          # Anything in padding should be masked
          if q >= length or k >= length:
            self.assertEqual(mask[b, 1, q, k], 0)
          # If the query is in the key
          elif q < prompt_length:
            # If the key is a previous position
            if k <= q:
              self.assertEqual(mask[b, 1, q, k], 1)
            # Everything else is masked, in input, the later prompt or padding.
            else:
              self.assertEqual(mask[b, 1, q, k], 0)
          # Past the prompt we can see everything, the earlier check handles
          # padding.
          elif q >= prompt_length:
            self.assertEqual(mask[b, 1, q, k], 1)

  @parameterized.named_parameters([
      dict(testcase_name="multitask", multitask=True),
      dict(testcase_name="single_task", multitask=False)
  ])
  def test_prompt_prompt_causal_and_inputs_attention(self, multitask):
    encoder_input_tokens, lengths = create_input(jnp.array([8, 1, 3, 4, 1, 5]))
    prompt_length = 4

    mask_fn = masks.prompt_encoder_attention_mask(
        prompt_length,
        {"prompt->prompt;causal", "prompt->input"},
        multitask=multitask,
    )
    mask = jax.jit(mask_fn)(encoder_input_tokens)

    if multitask:
      lengths = lengths - 1

    for b in range(mask.shape[0]):
      length = lengths[b] + prompt_length
      for q in range(mask.shape[2]):
        for k in range(mask.shape[3]):
          # Anything in padding should be masked
          if q >= length or k >= length:
            self.assertEqual(mask[b, 1, q, k], 0)
          elif q < prompt_length:
            if k <= q:
              self.assertEqual(mask[b, 1, q, k], 1)
            elif k >= prompt_length:
              self.assertEqual(mask[b, 1, q, k], 1)
            else:
              self.assertEqual(mask[b, 1, q, k], 0)
          elif q >= prompt_length:
            self.assertEqual(mask[b, 1, q, k], 1)

  @parameterized.product(
      dtype=[jnp.bfloat16, jnp.float32],
      attentions=[{"prompt->input"}, {"prompt->prompt"},
                  set()],
      multitask=[True, False])
  def test_mask_dtype(self, dtype, attentions, multitask):
    encoder_input_tokens, _ = create_input(jnp.array([5, 6, 7, 6, 1]))
    prompt_length = 4

    mask_fn = masks.prompt_encoder_attention_mask(
        prompt_length,
        attentions,
        multitask=multitask,
    )
    mask_fn = functools.partial(mask_fn, dtype=dtype)
    mask = jax.jit(mask_fn)(encoder_input_tokens)

    self.assertEqual(mask.dtype, dtype)

  @parameterized.named_parameters([
      dict(testcase_name="multitask", multitask=True),
      dict(testcase_name="single_task", multitask=False)
  ])
  def test_mask_shape(self, multitask):
    encoder_input_tokens, lengths = create_input(jnp.array([7, 12, 10, 3]))
    prompt_length = 13

    mask_fn = masks.prompt_encoder_attention_mask(
        prompt_length, set(), multitask=multitask)
    mask = jax.jit(mask_fn)(encoder_input_tokens)

    length = prompt_length + jnp.max(lengths)
    if multitask:
      length = length - 1
    self.assertEqual(mask.shape, (lengths.shape[0], 1, length, length))


if __name__ == "__main__":
  absltest.main()

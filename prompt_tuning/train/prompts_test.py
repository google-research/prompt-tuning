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

"""Tests for training prompts."""

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from prompt_tuning import test_utils
from prompt_tuning.train import prompts as train_prompts


class PromptsTest(parameterized.TestCase):

  def test_prefix_prompt(self):
    embed_size = 20
    prompt_length = 5
    batch_size = 2
    seq_len = 14
    prompt = jnp.zeros((batch_size, prompt_length, embed_size))
    embed = jnp.ones((batch_size, seq_len, embed_size))
    with_prompt = jax.jit(train_prompts.prefix_prompt)(prompt, embed, None)
    self.assertEqual(with_prompt.shape,
                     (batch_size, seq_len + prompt_length, embed_size))
    for i, example in enumerate(with_prompt):
      np.testing.assert_array_equal(example[:prompt_length], prompt[i])
      np.testing.assert_array_equal(example[prompt_length:], embed[i])

  def test_prefix_prompt_after_bos(self):
    embed_size = 20
    prompt_length = 5
    batch_size = 2
    seq_len = 14
    prompt = jnp.ones((batch_size, prompt_length, embed_size))
    embed = jnp.concatenate(
        [jnp.zeros((batch_size, 1, embed_size)),
         jnp.full((batch_size, seq_len - 1, embed_size), 2)],
        axis=1)
    with_prompt = jax.jit(train_prompts.prefix_prompt_after_bos)(prompt,
                                                                 embed,
                                                                 None)
    self.assertEqual(with_prompt.shape,
                     (batch_size, seq_len + prompt_length, embed_size))
    for i, example in enumerate(with_prompt):
      np.testing.assert_array_equal(example[1:prompt_length + 1], prompt[i])
      np.testing.assert_array_equal(example[prompt_length + 1:], embed[i, 1:])
      np.testing.assert_equal(np.asarray(example[0]), 0)

  @parameterized.product(
      decoder_only=[True, False],
      before_eos=[True, False]
  )
  def test_suffix_prompt(self, decoder_only, before_eos):
    embed_size = 20
    prompt_length = 5
    lengths = [14, 7]
    batch_size = 2
    seq_len = 14
    padding_embed_value = -1
    eos_value = -2
    prompt = jnp.reshape(jnp.arange(batch_size * prompt_length * embed_size),
                         (batch_size, prompt_length, embed_size))
    embed = np.ones((batch_size, seq_len, embed_size))
    inputs = np.ones((batch_size, seq_len))
    for i, l in enumerate(lengths):
      inputs[i, l:] = 0
      embed[i, l:, :] = padding_embed_value
      embed[i, l - 1, :] = eos_value
    if decoder_only:
      inputs[:, 0] = 0
    inputs = jnp.asarray(inputs)
    embed = jnp.asarray(embed)
    with_prompt = jax.jit(train_prompts.suffix_prompt, static_argnums=(3, 4))(
        prompt, embed, inputs, decoder_only=decoder_only, before_eos=before_eos)
    self.assertEqual(with_prompt.shape,
                     (batch_size, seq_len + prompt_length, embed_size))
    for i, (example, l) in enumerate(zip(with_prompt, lengths)):
      if before_eos:
        np.testing.assert_array_equal(example[:l - 1], embed[i, :l - 1])
        np.testing.assert_array_equal(example[l + prompt_length], embed[i, l])
        np.testing.assert_array_equal(example[l - 1:l - 1 + prompt_length],
                                      prompt[i])
      else:
        np.testing.assert_array_equal(example[:l], embed[i, :l])
        np.testing.assert_array_equal(example[l:l + prompt_length], prompt[i])
      padding = example[l + prompt_length:]
      np.testing.assert_array_equal(padding,
                                    np.full_like(padding, padding_embed_value))

  def test_prompt_does_concatenation(self):
    embed_size = 20
    prompt_length = 5
    batch_size = 2
    seq_len = 20
    mock_prompt = mock.MagicMock()
    prompt = jnp.zeros((prompt_length, embed_size))
    mock_prompt.return_value = prompt
    mock_combine = mock.create_autospec(
        train_prompts.prefix_prompt, spec_set=True)
    prompt_module = train_prompts.Prompt(
        prompt=mock_prompt, combine=mock_combine)
    input_tokens = jnp.ones((batch_size, seq_len))
    embed = jnp.ones((batch_size, seq_len, embed_size))
    prompt_module.apply({"params": {}}, input_tokens, embed)
    expanded_prompt = jnp.zeros((batch_size, prompt_length, embed_size))

    mock_prompt.assert_called_once_with(
        test_utils.ArrayEqualMatcher(input_tokens),
        test_utils.ArrayAllCloseMatcher(embed))
    mock_combine.assert_called_once_with(
        test_utils.ArrayAllCloseMatcher(expanded_prompt),
        test_utils.ArrayAllCloseMatcher(embed),
        test_utils.ArrayEqualMatcher(input_tokens))


if __name__ == "__main__":
  absltest.main()

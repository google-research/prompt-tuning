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

"""Tests for per layer prompting."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from prompt_tuning.extended.train import per_layer


class PerLayerPromptsTest(absltest.TestCase):

  def test_replace_prompt(self):
    embed_size = 20
    prompt_length = 5
    batch_size = 2
    seq_len = 14
    prompt = jnp.zeros((batch_size, prompt_length, embed_size))
    embed = jnp.ones((batch_size, seq_len, embed_size))
    with_prompt = jax.jit(per_layer.replace_prompt)(prompt, embed, None)
    self.assertEqual(with_prompt.shape, embed.shape)
    np.testing.assert_array_equal(with_prompt[:, :prompt_length], prompt)
    np.testing.assert_array_equal(with_prompt[:, prompt_length:],
                                  embed[:, prompt_length:])

  def test_add_prompt(self):
    embed_size = 20
    prompt_length = 5
    batch_size = 2
    seq_len = 14
    prompt = 2 * jnp.ones((batch_size, prompt_length, embed_size))
    embed = jnp.ones((batch_size, seq_len, embed_size))
    with_prompt = jax.jit(per_layer.add_prompt)(prompt, embed, None)
    self.assertEqual(with_prompt.shape, embed.shape)
    prompt_from_output = with_prompt[:, :prompt_length]
    np.testing.assert_array_equal(prompt_from_output,
                                  3 * jnp.ones_like(prompt_from_output))
    np.testing.assert_array_equal(with_prompt[:, prompt_length:],
                                  embed[:, prompt_length:])


if __name__ == "__main__":
  absltest.main()

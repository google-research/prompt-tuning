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

"""Tests for ia3.py."""

from absl.testing import absltest
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from prompt_tuning.extended.train import ia3


class IA3Test(absltest.TestCase):

  def test_reshpaed_attention_equal_default(self):
    batch_size = 12
    seq = 10
    heads = 3
    kv = 2
    hidden = heads * kv

    default_input = jax.random.uniform(jax.random.PRNGKey(0),
                                       (batch_size, seq, hidden))
    attention_input = jnp.reshape(default_input, (batch_size, seq, heads, kv))

    params = ia3.IA3(ia3_init=nn.initializers.uniform()).init(
        jax.random.PRNGKey(0), default_input)
    attention_params = jax.tree_map(
        lambda x: jnp.reshape(x, (heads, kv)), params)

    default_result = jax.jit(ia3.IA3().apply)(params, default_input)
    attention_result = jax.jit(ia3.IA3Attention().apply)(attention_params,
                                                         attention_input)
    np.testing.assert_allclose(jnp.reshape(default_result, (-1,)),
                               jnp.reshape(attention_result, (-1,)))


if __name__ == "__main__":
  absltest.main()

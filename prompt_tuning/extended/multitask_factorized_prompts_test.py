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

"""Tests for multi-task factorized prompts."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from prompt_tuning.extended import multitask_factorized_prompts as mfp


BATCH_SIZE = 10
LANG_LEN = 6
TASK_LEN = 4
EMBED_DIM = 20


class MultiTaskFactorizedPromptTest(parameterized.TestCase):

  def test_concat_langs_and_tasks_shapes(self):
    langs = jnp.reshape(
        jnp.arange(BATCH_SIZE * LANG_LEN * EMBED_DIM),
        (BATCH_SIZE, LANG_LEN * EMBED_DIM))
    tasks = jnp.reshape(
        jnp.arange(BATCH_SIZE * TASK_LEN * EMBED_DIM),
        (BATCH_SIZE, TASK_LEN * EMBED_DIM))

    concat = jax.jit(mfp.concat_langs_and_tasks)(langs, tasks)
    self.assertLen(concat.shape, 3)
    self.assertEqual(concat.shape, (BATCH_SIZE, LANG_LEN + TASK_LEN, EMBED_DIM))

  def test_concat_langs_and_tasks_values(self):
    langs = jnp.reshape(
        jnp.arange(BATCH_SIZE * LANG_LEN * EMBED_DIM),
        (BATCH_SIZE, LANG_LEN * EMBED_DIM))
    tasks = jnp.reshape(
        jnp.arange(BATCH_SIZE * TASK_LEN * EMBED_DIM),
        (BATCH_SIZE, TASK_LEN * EMBED_DIM))

    langs = jnp.reshape(langs, (BATCH_SIZE, LANG_LEN, EMBED_DIM))
    tasks = jnp.reshape(tasks, (BATCH_SIZE, TASK_LEN, EMBED_DIM))

    concat = jax.jit(mfp.concat_langs_and_tasks)(langs, tasks)
    for i, c in enumerate(concat):
      np.testing.assert_allclose(c[:LANG_LEN, :], langs[i])
      np.testing.assert_allclose(c[LANG_LEN:, :], tasks[i])


if __name__ == "__main__":
  absltest.main()

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

"""Tests for multi-task prompts."""

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from prompt_tuning.extended import multitask_prompts as prompts


BATCH_SIZE = 10
SEQ_LEN = 5
SEQ_LEN2 = 4
FEATURES = 20
FEATURES2 = 7


class MultiTaskPromptTest(parameterized.TestCase):

  @parameterized.parameters([1, 3.4, "example", None])
  def test_identity(self, value):
    self.assertEqual(prompts.identity(value), value)
    self.assertIs(prompts.identity(value), value)

  @parameterized.parameters([1, 3.4])
  def test_identity_jited(self, value):
    self.assertEqual(jax.jit(prompts.identity)(value), value)

  def test_identity_jnp(self):
    value = jnp.ones((4, 6))
    np.testing.assert_array_equal(jax.jit(prompts.identity)(value), value)

  def test_add_shared_and_tasks_shapes(self):
    shared = jnp.reshape(jnp.arange(SEQ_LEN * FEATURES), (SEQ_LEN, FEATURES))
    specific = jnp.reshape(
        jnp.arange(BATCH_SIZE * SEQ_LEN * FEATURES),
        (BATCH_SIZE, SEQ_LEN * FEATURES))
    added = jax.jit(prompts.add_shared_and_tasks)(shared, specific)
    self.assertLen(added.shape, 3)
    self.assertEqual(added.shape, (BATCH_SIZE, SEQ_LEN, FEATURES))

  def test_add_shared_and_tasks_values(self):
    shared = jnp.reshape(jnp.arange(SEQ_LEN * FEATURES), (SEQ_LEN, FEATURES))
    specific = jnp.reshape(
        jnp.arange(BATCH_SIZE * SEQ_LEN * FEATURES),
        (BATCH_SIZE, SEQ_LEN * FEATURES))
    specific_prompt = jnp.reshape(specific, (BATCH_SIZE, SEQ_LEN, FEATURES))
    added = jax.jit(prompts.add_shared_and_tasks)(shared, specific)

    for i, add in enumerate(added):
      np.testing.assert_allclose(add, shared + specific_prompt[i])

  def test_concat_shared_and_tasks_shapes(self):
    shared = jnp.reshape(jnp.arange(SEQ_LEN * FEATURES), (SEQ_LEN, FEATURES))
    specific = jnp.reshape(
        jnp.arange(BATCH_SIZE * SEQ_LEN2 * FEATURES),
        (BATCH_SIZE, SEQ_LEN2 * FEATURES))

    concat = jax.jit(prompts.concat_shared_and_tasks)(shared, specific)
    self.assertLen(concat.shape, 3)
    self.assertEqual(concat.shape, (BATCH_SIZE, SEQ_LEN + SEQ_LEN2, FEATURES))

  def test_concat_shared_and_tasks_values(self):
    shared = jnp.reshape(jnp.arange(SEQ_LEN * FEATURES), (SEQ_LEN, FEATURES))
    specific = jnp.reshape(
        jnp.arange(BATCH_SIZE * SEQ_LEN2 * FEATURES),
        (BATCH_SIZE, SEQ_LEN2 * FEATURES))
    specific_prompt = jnp.reshape(specific, (BATCH_SIZE, SEQ_LEN2, FEATURES))

    concat = jax.jit(prompts.concat_shared_and_tasks)(shared, specific)
    for i, c in enumerate(concat):
      np.testing.assert_allclose(c[:SEQ_LEN, :], shared)
      np.testing.assert_allclose(c[SEQ_LEN:, :], specific_prompt[i])

  def test_concat_features_shared_and_tasks_shapes(self):
    shared = jnp.reshape(jnp.arange(SEQ_LEN * FEATURES), (SEQ_LEN, FEATURES))
    specific = jnp.reshape(
        jnp.arange(BATCH_SIZE * SEQ_LEN * FEATURES2),
        (BATCH_SIZE, SEQ_LEN * FEATURES2))

    concat = jax.jit(prompts.concat_features_shared_and_tasks)(shared, specific)
    self.assertLen(concat.shape, 3)
    self.assertEqual(concat.shape, (BATCH_SIZE, SEQ_LEN, FEATURES + FEATURES2))

  def test_concat_features_shared_and_tasks_values(self):
    shared = jnp.reshape(jnp.arange(SEQ_LEN * FEATURES), (SEQ_LEN, FEATURES))
    specific = jnp.reshape(
        jnp.arange(BATCH_SIZE * SEQ_LEN * FEATURES2),
        (BATCH_SIZE, SEQ_LEN * FEATURES2))
    specific_prompt = jnp.reshape(specific, (BATCH_SIZE, SEQ_LEN, FEATURES2))

    concat = jax.jit(prompts.concat_features_shared_and_tasks)(shared, specific)
    for i, c in enumerate(concat):
      np.testing.assert_allclose(c[:, :FEATURES], shared)
      np.testing.assert_allclose(c[:, FEATURES:], specific_prompt[i])

  def test_project_shared_and_tasks_shape(self):
    shared = jnp.reshape(jnp.arange(SEQ_LEN * FEATURES), (SEQ_LEN, FEATURES))
    kernels = [
        jnp.reshape(
            jnp.arange(FEATURES * FEATURES2) + i, (FEATURES2, FEATURES))
        for i in range(BATCH_SIZE)
    ]
    bias = [jnp.arange(FEATURES2) + i for i in range(BATCH_SIZE)]

    flat_kernels = jnp.stack([jnp.ravel(k) for k in kernels])
    flat_bias = jnp.stack(bias)
    specific = jnp.concatenate([flat_kernels, flat_bias], axis=1)

    proj = jax.jit(prompts.project_shared_with_tasks)(shared, specific)
    self.assertLen(proj.shape, 3)
    self.assertEqual(proj.shape, (BATCH_SIZE, SEQ_LEN, FEATURES2))

  def test_project_shared_and_tasks_values(self):
    shared = jnp.reshape(jnp.arange(SEQ_LEN * FEATURES), (SEQ_LEN, FEATURES))
    kernels = [
        jnp.reshape(
            jnp.arange(FEATURES * FEATURES2) + i, (FEATURES2, FEATURES))
        for i in range(BATCH_SIZE)
    ]
    bias = [jnp.arange(FEATURES2) + i for i in range(BATCH_SIZE)]

    flat_kernels = jnp.stack([jnp.ravel(k) for k in kernels])
    flat_bias = jnp.stack(bias)
    specific = jnp.concatenate([flat_kernels, flat_bias], axis=1)

    proj = jax.jit(prompts.project_shared_with_tasks)(shared, specific)

    for i, p in enumerate(proj):
      np.testing.assert_allclose(p, shared @ kernels[i].T + bias[i])

  @parameterized.parameters([8, 100], [100, 32100])
  def test_sentinels_to_task_index(self, num_tasks, vocab_size):
    start = vocab_size - num_tasks
    sentinels = jnp.arange(start, vocab_size)[::-1]
    task_idx = jax.jit(
        prompts.sentinel_to_task, static_argnums=1)(sentinels, vocab_size)
    np.testing.assert_allclose(task_idx, jnp.arange(num_tasks))

  def test_multi_task_lookup_tasks(self):
    num_tasks = 8
    length = 5
    task_size = 3
    gold_tasks = np.tile(
        np.reshape(np.arange(num_tasks), (num_tasks, 1)), (1, task_size))
    prompt = prompts.MultiTaskPrompt(
        length=length,
        num_tasks=num_tasks,
        shared_init=nn.initializers.zeros,
        tasks_init=lambda *args, **kwargs: gold_tasks,
        get_shared_features=lambda _, x: x,
        get_task_size=lambda _, x, y: task_size,
        combine_shared_and_tasks=lambda _, tasks: tasks,
    )
    params = prompt.init(
        jax.random.PRNGKey(0), jnp.zeros((2, 3), dtype=jnp.int32),
        jnp.zeros((2, 3, 3), dtype=jnp.int32))
    input_ = jnp.arange(num_tasks)
    input_ = jax.random.permutation(jax.random.PRNGKey(0),
                                    input_,
                                    independent=True)
    tasks = jax.jit(prompt.apply)(params, jnp.reshape(input_, (-1, 1)),
                                  jnp.zeros((num_tasks, 1, task_size)))
    for task, idx in zip(tasks, input_):
      np.testing.assert_array_equal(task, idx)

  def test_input_features(self):
    x_embed_features = 512
    shared_features = prompts.input_features(None, x_embed_features)
    self.assertEqual(shared_features, x_embed_features)

  def test_shared_features_attr_or_input_features_no_attr(self):
    input_features = 256
    prompt = mock.create_autospec(
        prompts.ConcatFeaturesMultiTaskPrompt,
        spec_set=True,
        instance=True,
        shared_features=None)

    shared_features = prompts.shared_features_attr_or_input_features(
        prompt, input_features)
    self.assertEqual(shared_features, input_features)

  def test_shared_features_attr_or_input_features_with_attr(self):
    features = 512
    input_features = 256

    prompt_with_shared = mock.create_autospec(
        prompts.ConcatFeaturesMultiTaskPrompt,
        spec_set=True,
        instance=True,
        shared_features=features)
    shared_features = prompts.shared_features_attr_or_input_features(
        prompt_with_shared, input_features)
    self.assertEqual(shared_features, features)

  def test_match_shared(self):
    shared_length = 20
    shared_features = 512
    gold = shared_length * shared_features

    task_size = prompts.match_shared(None, shared_length, shared_features)
    self.assertEqual(task_size, gold)

  def test_task_length_attr_or_match_shared_no_attr(self):
    shared_length = 20
    shared_features = 512
    prompt = mock.create_autospec(
        prompts.ConcatMultiTaskPrompt,
        spec_set=True,
        instance=True,
        task_length=None)
    gold = shared_length * shared_features

    task_size = prompts.task_length_attr_or_match_shared(
        prompt, shared_length, shared_features)
    self.assertEqual(task_size, gold)

  def test_task_length_attr_or_match_shared(self):
    shared_length = 20
    shared_features = 512
    task_length = 10
    prompt_with_task_length = mock.create_autospec(
        prompts.ConcatMultiTaskPrompt,
        spec_set=True,
        instance=True,
        task_length=task_length)
    gold = task_length * shared_features

    task_size = prompts.task_length_attr_or_match_shared(
        prompt_with_task_length, shared_length, shared_features)
    self.assertEqual(task_size, gold)

  def test_task_features_or_match_shared_no_attr(self):
    shared_length = 30
    shared_features = 768
    prompt = mock.create_autospec(
        prompts.ConcatFeaturesMultiTaskPrompt,
        spec_set=True,
        instance=True,
        task_features=None)
    gold = shared_length * shared_features

    task_size = prompts.task_features_or_match_shared(prompt, shared_length,
                                                      shared_features)
    self.assertEqual(task_size, gold)

  def test_task_features_or_match_shared(self):
    shared_length = 30
    shared_features = 768
    task_features = 256
    prompt_with_features = mock.create_autospec(
        prompts.ConcatFeaturesMultiTaskPrompt,
        spec_set=True,
        instance=True,
        task_features=task_features)
    gold = shared_length * task_features

    task_size = prompts.task_features_or_match_shared(prompt_with_features,
                                                      shared_length,
                                                      shared_features)
    self.assertEqual(task_size, gold)

  def test_projection_size(self):
    shared_length = None
    shared_features = 128
    task_features = 256
    prompt = mock.create_autospec(
        prompts.ProjectMultiTaskPrompt,
        spec_set=True,
        instance=True,
        task_features=task_features)
    kernel_gold = shared_features * task_features
    bias_gold = task_features
    gold = kernel_gold + bias_gold

    task_size = prompts.projection_size(prompt, shared_length, shared_features)
    self.assertEqual(task_size, gold)

  def test_projection_default_features(self):
    shared_length = None
    shared_features = 128
    prompt = mock.create_autospec(
        prompts.ProjectMultiTaskPrompt,
        spec_set=True,
        instance=True,
        task_features=None)
    kernel_gold = shared_features * shared_features
    bias_gold = shared_features
    gold = kernel_gold + bias_gold

    task_size = prompts.projection_size(prompt, shared_length, shared_features)
    self.assertEqual(task_size, gold)


if __name__ == "__main__":
  absltest.main()

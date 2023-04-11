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

"""Tests for prompts."""

import os
from typing import List
from unittest import mock
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from prompt_tuning import prompts
import seqio
from tensorflow.io import gfile

FLAGS = flags.FLAGS
TEST_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "test_data")

BATCH_SIZE = 10
SEQ_LEN = 5
FEATURES = 20


def load_numpy(test_data):
  """Load a file with the np_load function."""
  path = os.path.join(test_data, "prompt_5x256.npy")
  return prompts.np_load(path)


def load_t5x(test_data):
  """Load a file with the t5x_load function."""
  checkpoint_path = os.path.join(test_data,
                                 "test_t5_1_1_tiny",
                                 "checkpoint_3")
  variable_path = "token_embedder/embedding"
  return prompts.t5x_load(checkpoint_path, variable_path)


class PromptsTest(parameterized.TestCase):

  def test_expand_shapes(self):
    x = jnp.reshape(jnp.arange(SEQ_LEN * FEATURES), (SEQ_LEN, FEATURES))
    y = jnp.ones((BATCH_SIZE, 1, 1))
    z = jax.jit(prompts.expand_to_batch)(x, y)
    self.assertLen(z.shape, len(x.shape) + 1)
    self.assertEqual(z.shape, (BATCH_SIZE, SEQ_LEN, FEATURES))

  def test_expand_values(self):
    x = jnp.reshape(jnp.arange(SEQ_LEN * FEATURES), (SEQ_LEN, FEATURES))
    y = jnp.ones((BATCH_SIZE, 1, 1))
    z = jax.jit(prompts.expand_to_batch)(x, y)

    for i in range(1, len(z)):
      np.testing.assert_allclose(z[i], z[0])

  @parameterized.named_parameters(
      dict(testcase_name="numpy",
           gold_file="prompt_5x256.npy",
           loader=load_numpy,
           shape=(5, 256)),
      dict(testcase_name="t5x",
           gold_file="tiny_embeddings_32128x4.npy",
           loader=load_t5x,
           shape=(32128, 4))
  )
  def test_from_array_numpy(self, gold_file, loader, shape):
    gold_file = os.path.join(TEST_DATA, gold_file)

    from_file = loader(TEST_DATA)

    file_initializer = prompts.from_array(from_file)

    with gfile.GFile(gold_file, "rb") as rf:
      gold_prompt = np.load(rf)

    np.testing.assert_allclose(
        jax.jit(file_initializer, static_argnums=1)(None, shape),
        gold_prompt)

  @parameterized.named_parameters(
      dict(testcase_name="numpy",
           loader=load_numpy),
      dict(testcase_name="t5x",
           loader=load_t5x)
  )
  def test_from_array_wrong_shape(self, loader):
    from_file = loader(TEST_DATA)
    file_initializer = prompts.from_array(from_file)
    wrong_shape = (256, 5)

    with self.assertRaises(ValueError):
      jax.jit(file_initializer, static_argnums=1)(None, wrong_shape)

  def test_from_embedded_list(self):
    prompt_length = 10
    vocab_size = 100
    embed_size = 10
    vocab_mock = mock.create_autospec(
        seqio.SentencePieceVocabulary, instance=True, spec_set=True)

    def fake_vocab_lookup(s: str) -> List[int]:
      if s == "true":
        return [23, 54]
      elif s == "false":
        return [45]
      return [0]

    vocab_mock.encode.side_effect = fake_vocab_lookup

    true_gold = np.full(embed_size, (23 + 54) / 2)
    false_gold = np.full(embed_size, 45)

    fake_embeddings = np.tile(
        np.reshape(np.arange(vocab_size), (-1, 1)), (1, embed_size))

    init = prompts.from_embedded_list(fake_embeddings, vocab_mock,
                                      ["true", "false"], nn.initializers.zeros)
    prompt = jax.jit(
        init, static_argnums=1)(jax.random.PRNGKey(0),
                                (prompt_length, embed_size))

    self.assertEqual(prompt.shape, (prompt_length, embed_size))
    np.testing.assert_equal(prompt[0, :], true_gold)
    np.testing.assert_equal(prompt[1, :], false_gold)
    self.assertTrue(np.all(np.equal(prompt[2, :], 0)))

  def test_from_embedded_list_longer_than_prompt_size(self):
    prompt_length = 2
    vocab_size = 100
    embed_size = 10
    vocab_mock = mock.create_autospec(
        seqio.SentencePieceVocabulary, instance=True, spec_set=True)

    def fake_vocab_lookup(s: str) -> List[int]:
      if s == "true":
        return [23, 54]
      elif s == "false":
        return [45]
      return [0]

    vocab_mock.encode.side_effect = fake_vocab_lookup

    true_gold = np.full(embed_size, (23 + 54) / 2)
    false_gold = np.full(embed_size, 45)

    fake_embeddings = np.tile(
        np.reshape(np.arange(vocab_size), (-1, 1)), (1, embed_size))

    init = prompts.from_embedded_list(fake_embeddings, vocab_mock,
                                      ["true", "false", "maybe"],
                                      nn.initializers.zeros)
    prompt = jax.jit(
        init, static_argnums=1)(jax.random.PRNGKey(0),
                                (prompt_length, embed_size))

    self.assertEqual(prompt.shape, (prompt_length, embed_size))
    np.testing.assert_equal(prompt[0, :], true_gold)
    np.testing.assert_equal(prompt[1, :], false_gold)

  def test_from_embedded_string_longer_than_prompt_size(self):
    prompt_length = 2
    vocab_size = 100
    embed_size = 10
    vocab_mock = mock.create_autospec(
        seqio.SentencePieceVocabulary, instance=True, spec_set=True)

    vocab_mock.encode.return_value = [10, 50, 75]

    fake_embeddings = np.tile(
        np.reshape(np.arange(vocab_size), (-1, 1)), (1, embed_size))

    gold = fake_embeddings[[10, 50,], :]

    init = prompts.from_embedded_string(fake_embeddings, vocab_mock,
                                        "this is my example",
                                        nn.initializers.zeros)

    prompt = jax.jit(
        init, static_argnums=1)(jax.random.PRNGKey(0),
                                (prompt_length, embed_size))

    self.assertEqual(prompt.shape, (prompt_length, embed_size))
    np.testing.assert_equal(prompt, gold)

  def test_from_embedded_string_shorter_than_prompt_size(self):
    prompt_length = 6
    vocab_size = 100
    embed_size = 10
    vocab_mock = mock.create_autospec(
        seqio.SentencePieceVocabulary, instance=True, spec_set=True)

    vocab_mock.encode.return_value = [10, 50, 75]

    fake_embeddings = np.tile(
        np.reshape(np.arange(vocab_size), (-1, 1)), (1, embed_size))

    gold = fake_embeddings[[10, 50, 75], :]

    init = prompts.from_embedded_string(fake_embeddings, vocab_mock,
                                        "this is my example",
                                        nn.initializers.zeros)

    prompt = jax.jit(
        init, static_argnums=1)(jax.random.PRNGKey(0),
                                (prompt_length, embed_size))

    self.assertEqual(prompt.shape, (prompt_length, embed_size))
    np.testing.assert_equal(prompt[:3], gold)
    np.testing.assert_equal(prompt[3:], np.zeros((3, embed_size)))

  @parameterized.named_parameters(
      dict(testcase_name="embedded_list",
           init_fn=prompts.from_embedded_list),
      dict(testcase_name="embedded_string",
           init_fn=prompts.from_embedded_string))
  def test_from_embedded_list_wrong_features(self, init_fn):
    vocab_size = 100
    embed_size = 10
    prompt_size = 20
    prompt_length = 15
    embeddings = jnp.zeros((vocab_size, embed_size))

    init = init_fn(embeddings, None, None, None)
    with self.assertRaises(ValueError):
      jax.jit(
          init, static_argnums=1)(jax.random.PRNGKey(0),
                                  (prompt_length, prompt_size))

  def test_from_sample_of_embeddings(self):
    vocab_size = 100
    embed_size = 10
    embeddings = np.tile(
        np.reshape(np.arange(vocab_size), (-1, 1)), (1, embed_size))
    with mock.patch.object(
        prompts.jax.random, "choice", autospec=True) as choice_mock:
      index = jnp.array([0, 24, 65, 78, 93, 50])
      choice_mock.return_value = index
      k = len(index)
      init = prompts.from_sample_of_embeddings(embeddings, k)
      prompt = jax.jit(
          init, static_argnums=1)(jax.random.PRNGKey(0), (k, embed_size))
      for p, i in zip(prompt, index):
        self.assertTrue(np.all(np.equal(p, i)))

  def test_from_sample_of_embeddings_shape(self):
    vocab_size = 100
    embed_size = 10
    embeddings = np.zeros((vocab_size, embed_size))
    prompt_length = 20
    init = prompts.from_sample_of_embeddings(embeddings, 100)
    prompt = jax.jit(
        init, static_argnums=1)(jax.random.PRNGKey(0),
                                (prompt_length, embed_size))
    self.assertEqual(prompt.shape, (prompt_length, embed_size))

  def test_from_sample_of_embeddings_wrong_feature(self):
    vocab_size = 100
    embed_size = 10
    prompt_size = 20
    prompt_length = 15
    embeddings = np.zeros((vocab_size, embed_size))

    init = prompts.from_sample_of_embeddings(embeddings, None)
    with self.assertRaises(ValueError):
      jax.jit(
          init, static_argnums=1)(jax.random.PRNGKey(0),
                                  (prompt_length, prompt_size))

  def test_from_sample_of_embeddings_no_population(self):
    vocab_size = 100
    embed_size = 10
    population_size = 0
    embeddings = jnp.zeros((vocab_size, embed_size))

    with self.assertRaises(ValueError):
      prompts.from_sample_of_embeddings(embeddings, population_size)

  def test_from_sample_of_embeddings_with_population_size(self):
    vocab_size = 100
    embed_size = 10
    population_size = 50
    prompt_length = 15
    embeddings = np.zeros((vocab_size, embed_size))

    init = prompts.from_sample_of_embeddings(embeddings, population_size)
    with mock.patch.object(
        prompts.jax.random, "choice", autospec=True) as choice_mock:
      choice_mock.return_value = jnp.zeros((prompt_length), dtype=jnp.int32)
      rng = jax.random.PRNGKey(0)
      # Because we are checking the mock value, we can't jit, otherwise we will
      # get an error about the tracer "leaking".
      init(rng, (prompt_length, embed_size))
      choice_mock.assert_called_once_with(rng, population_size,
                                          (prompt_length,), False)

  @parameterized.named_parameters([
      dict(testcase_name="None", population_size=None),
      dict(testcase_name="too_big", population_size=1000)
  ])
  def test_from_sample_of_embeddings_population_size_mismatch(
      self, population_size):
    vocab_size = 100
    embed_size = 10
    prompt_length = 15
    embeddings = np.zeros((vocab_size, embed_size))

    init = prompts.from_sample_of_embeddings(embeddings, population_size)
    with mock.patch.object(
        prompts.jax.random, "choice", autospec=True) as choice_mock:
      choice_mock.return_value = jnp.zeros((prompt_length), dtype=jnp.int32)
      rng = jax.random.PRNGKey(0)
      init(rng, (prompt_length, embed_size))
      choice_mock.assert_called_once_with(rng, vocab_size, (prompt_length,),
                                          False)

  def test_from_sample_of_embeddings_population_too_small(self):
    vocab_size = 100
    embed_size = 10
    population_size = 5
    prompt_length = 15
    embeddings = np.zeros((vocab_size, embed_size))

    init = prompts.from_sample_of_embeddings(embeddings, population_size)
    with mock.patch.object(
        prompts.jax.random, "choice", autospec=True) as choice_mock:
      choice_mock.return_value = jnp.zeros((prompt_length), dtype=jnp.int32)
      rng = jax.random.PRNGKey(0)
      init(rng, (prompt_length, embed_size))
      choice_mock.assert_called_once_with(rng, population_size,
                                          (prompt_length,), True)

  @parameterized.parameters([jnp.float32, jnp.bfloat16])
  def test_inference_only_prompt(self, dtype):
    length = 5
    embed_dim = 256
    prompt_file = os.path.join(TEST_DATA, "prompt_5x256.npy")
    prompt_init = prompts.from_array(prompts.np_load(prompt_file))

    prompt_module = prompts.InferenceOnlyPrompt(
        length, embed_dim, prompt_init, dtype)
    _ = prompt_module.init(
        jax.random.PRNGKey(0), jnp.zeros((1, 2)), jnp.zeros((1, 2, 5)))
    prompt = jax.jit(prompt_module.apply)(
        {}, jnp.zeros((1, 2)), jnp.zeros((1, 2, 5)))

    prompt_from_file = jnp.array(load_numpy(TEST_DATA)).astype(dtype)
    self.assertEqual(prompt.shape, (length, embed_dim))
    self.assertEqual(prompt.dtype, dtype)
    # Cast all to float32 as bfloat16 was causing issues in the allclose.
    np.testing.assert_allclose(prompt.astype(jnp.float32),
                               prompt_from_file.astype(jnp.float32))


if __name__ == "__main__":
  absltest.main()

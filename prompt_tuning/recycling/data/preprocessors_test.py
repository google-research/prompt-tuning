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

"""Tests for prompt recycling preprocessors."""

import collections
import itertools
import os
from absl.testing import absltest
import numpy as np
from prompt_tuning.recycling.data import preprocessors
import seqio
import t5.data
import tensorflow as tf
import tensorflow_datasets as tfds

TRIALS = 400_000

# We need to go up 3 dirs to get to the test data path.
TEST_DATA = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)))),
    "test_data")


class RandomLabelsTest(absltest.TestCase):

  def _collect_stats_from_dataset(self, dataset):
    counts = collections.Counter(
        [tuple(ex["targets"].tolist()) for ex in tfds.as_numpy(dataset)])
    norm = sum(counts.values())
    return {k: v / norm for k, v in counts.most_common()}

  def test_uniform(self):
    targets = ["positive", "negative", "neither"]
    vocab = seqio.vocabularies.SentencePieceVocabulary(
        os.path.join(TEST_DATA, "t5_vocab"))

    ds = tf.data.Dataset.range(TRIALS)
    ds = ds.map(lambda ex: {"inputs": ex, "targets": None})
    ds = preprocessors.random_targets(
        ds,
        {"targets": t5.data.Feature(vocabulary=vocab)},
        None,
        targets,
        seed=42)

    stats = self._collect_stats_from_dataset(ds)
    for dist1, dist2 in itertools.product(stats.values(), stats.values()):
      np.testing.assert_allclose(dist1, dist2, rtol=1e-2)

  def test_skewed(self):
    targets = ["positive", "negative", "neither"]
    probs = [0.1, 0.3, 0.6]

    vocab = seqio.vocabularies.SentencePieceVocabulary(
        os.path.join(TEST_DATA, "t5_vocab"))

    ds = tf.data.Dataset.range(TRIALS)
    ds = ds.map(lambda ex: {"inputs": ex, "targets": None})
    ds = preprocessors.random_targets(
        ds,
        {"targets": t5.data.Feature(vocabulary=vocab)},
        None,
        targets,
        probs=probs,
        seed=42)

    stats = self._collect_stats_from_dataset(ds)
    for target, prob in zip(targets, probs):
      targ = tuple(vocab.encode(target))
      np.testing.assert_allclose(stats[targ], prob, rtol=1e-2)


if __name__ == "__main__":
  absltest.main()

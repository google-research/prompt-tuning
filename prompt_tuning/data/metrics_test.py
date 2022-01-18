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

"""Tests for metrics."""

import unittest.mock as mock
from absl.testing import absltest
from prompt_tuning.data import metrics


class MetricsTest(absltest.TestCase):

  def test_metric_with_examples(self):
    gold = "my_gold_value"
    metric_fn = mock.MagicMock(return_value=gold)
    targets = list(range(0, 100))
    predictions = list(range(100, 0, -1))
    prediction_field = "my_prediction_field"
    prediction_wrappers = [{prediction_field: p} for p in predictions]

    metric = metrics.metric_with_examples(metric_fn, targets,
                                          prediction_wrappers, prediction_field)
    self.assertEqual(metric, gold)
    metric_fn.assert_called_once_with(targets, predictions)

  @mock.patch("prompt_tuning.data.metrics.random.seed", autospec=True)
  def test_safe_sample_seeded(self, seed_mock):
    seed = 100
    metrics.safe_sample(12, [], seed=seed)
    seed_mock.assert_called_once_with(seed)

  def test_safe_sample_full(self):
    pop = [1, 2, 3, 4, 5, 10]
    result = metrics.safe_sample(-1, pop, None)
    self.assertEqual(result, range(len(pop)))

  def test_safe_sample_extra(self):
    pop = [4, 5, 6, 7, 8, 9, 100, -1]
    result = metrics.safe_sample(len(pop) + 12, pop, None)
    self.assertEqual(result, range(len(pop)))

  @mock.patch("prompt_tuning.data.metrics.random.sample", autospec=True)
  def test_safe_sample(self, sample_mock):
    pop = list("This is my population lol")
    k = 3
    gold = "lol"

    sample_mock.return_value = gold
    result = metrics.safe_sample(k, pop, None)
    self.assertEqual(result, gold)
    sample_mock.assert_called_once_with(range(len(pop)), k=k)

  def test_safe_sample_k(self):
    pop = list("This is my population lol")
    k = 5

    result = metrics.safe_sample(k, pop, None)
    self.assertLen(result, k)


if __name__ == "__main__":
  absltest.main()

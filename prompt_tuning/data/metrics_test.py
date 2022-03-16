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

import ast
import itertools
import json
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

  def test_metric_label_set_stats_distributions(self):
    gold_counts = {"1": 1, "2": 2, "3": 1, "4": 10, "5": 3}
    pred_counts = {"1": 2, "2": 2, "3": 3, "4": 5, "5": 5}
    self.assertEqual(sum(gold_counts.values()), sum(pred_counts.values()))
    gold_labels = list(itertools.chain(*([k] * v
                                         for k, v in gold_counts.items())))
    pred_labels = list(itertools.chain(*([k] * v
                                         for k, v in pred_counts.items())))
    predicted_labels = [{"prediction_pretokenized": p,
                         "targets_pretokenized": t}
                        for p, t in zip(pred_labels, gold_labels)]
    result = metrics.label_set_stats(
        None, predicted_labels, "m", normalize=False
    )["m label distributions"].textdata
    result = result.strip("`\n")
    result = json.loads(result)
    self.assertEqual(result["target label distribution"], gold_counts)
    self.assertEqual(result["prediction label distribution"], pred_counts)

  def test_metric_label_set_stats_distributions_normalized(self):
    gold_dist = {"1": 0.1, "2": 0.2, "3": 0.1, "4": 0.4, "5": 0.2}
    pred_dist = {"1": 0.2, "2": 0.2, "3": 0.05, "4": 0.25, "5": 0.3}
    total_length = 20
    self.assertEqual(sum(gold_dist.values()), 1.0)
    self.assertEqual(sum(gold_dist.values()), sum(pred_dist.values()))
    gold_labels = list(itertools.chain(*([k] * int(v * total_length)
                                         for k, v in gold_dist.items())))
    pred_labels = list(itertools.chain(*([k] * int(v * total_length)
                                         for k, v in pred_dist.items())))
    predicted_labels = [{"prediction_pretokenized": p,
                         "targets_pretokenized": t}
                        for p, t in zip(pred_labels, gold_labels)]
    result = metrics.label_set_stats(
        None, predicted_labels, "m", normalize=True
    )["m label distributions"].textdata
    result = result.strip("`\n")
    result = json.loads(result)
    self.assertEqual(result["target label distribution"], gold_dist)
    self.assertEqual(result["prediction label distribution"], pred_dist)

  def test_metric_label_set_stats(self):
    predicted_labels = [{"prediction_pretokenized": p,
                         "targets_pretokenized": t}
                        for p, t in zip(list("12345"), list("23462"))]
    target_card = 4
    pred_card = 5
    overlap = 3
    target_minus_pred = 1
    pred_minus_target = 2
    golds = (target_card,
             pred_card,
             overlap,
             target_minus_pred,
             pred_minus_target)
    result = metrics.label_set_stats(
        None, predicted_labels, "m")["m label stats"].textdata
    for line, metric in zip(result.split("\n\n"), golds):
      self.assertEqual(int(line.rsplit(":", maxsplit=1)[1]), metric)

  def test_metric_label_set_stats_with_sets(self):
    predicted_labels = [{"prediction_pretokenized": p,
                         "targets_pretokenized": t}
                        for p, t in zip(list("12545"), list("13462"))]
    target_card = 5
    target_set = {"1", "2", "3", "4", "6"}
    pred_card = 4
    pred_set = {"1", "2", "4", "5"}
    overlap = 3
    overlap_set = {"1", "2", "4"}
    target_minus_pred = 2
    t_m_p_set = {"3", "6"}
    pred_minus_target = 1
    p_m_t_set = {"5"}
    golds = (target_card, target_set,
             pred_card, pred_set,
             overlap, overlap_set,
             target_minus_pred, t_m_p_set,
             pred_minus_target, p_m_t_set)
    result = metrics.label_set_stats(
        None,
        predicted_labels,
        "m",
        display_sets=True)["m label stats"].textdata
    for line, metric in zip(result.split("\n\n"), golds):
      self.assertEqual(
          ast.literal_eval(line.rsplit(":", maxsplit=1)[1].strip()), metric)


if __name__ == "__main__":
  absltest.main()

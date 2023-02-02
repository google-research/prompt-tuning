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

"""A metric wrapper that updates the metric names.

This wrapper allows us to have two 2 rank classification metrics for a single
task with different settings (one with length normalization on and one off).
Otherwise we would need to create two copies of the task which would cause a
large slow-down in evaluation time as inference would be performed on each task,
even through they have the same inputs, only the metric calculation changes. The
wrapper enables us re-use the model's inference for both version of the metric.
"""

from typing import Union, Dict, Sequence, Tuple, Callable

RankClassificationTargets = Sequence[Tuple[Sequence[int], bool, float, int]]
RankClassificationScores = Sequence[float]
MetricMap = Dict[str, Union[float, int]]
RankClassificationMetric = Callable[
    [RankClassificationTargets, RankClassificationScores], MetricMap]


# Note: The parameter names need to be `target` and `scores` as T5X uses
# introspection to determine if it should use the autoregressive or the scoring
# based inference function.
def prefix_metric_names(
    targets: RankClassificationTargets,
    scores: RankClassificationScores,
    metric_prefix: str,
    metric_fn: RankClassificationMetric) -> MetricMap:
  """Run `metric_fn` and prepend `metric_prefix` to all metric names."""
  metrics = metric_fn(targets, scores)
  return {f"{metric_prefix}_{k}": v for k, v in metrics.items()}

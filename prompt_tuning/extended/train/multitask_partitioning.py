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

"""Custom, default partitioning rules for multitask prompt based models."""

from prompt_tuning.train import partitioning as pt_partitioning
from t5x import partitioning


def standard_logical_axis_rules() -> partitioning.LogicalAxisRules:
  """Add multitask prompt partitioning rules."""
  return pt_partitioning.standard_logical_axis_rules() + (
      ("tasks", None), ("prompt+embed", None))

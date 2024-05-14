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

"""A collection of the default features used in various t5 tasks."""

try:
  # This makes sure we have access to the byt5 mixtures
  import byt5.tasks as byt5_tasks
except ImportError:
  byt5_tasks = None
try:
  # This makes sure we have access to all mt5 tasks and mixtures
  import multilingual_t5.tasks as mt5_tasks
except ImportError:
  mt5_tasks = None
# This makes sure we have access to all the t5 mixtures
import t5.data.mixtures  # pylint: disable=unused-import
# This makes sure we have access to all the t5 tasks
import t5.data.tasks as t5_tasks

# Aliases for ease of use when you want features for a specific model.
T5_FEATURES = t5_tasks.DEFAULT_OUTPUT_FEATURES
MODEL_TO_FEATURES = {
    # We use the keys to prefix the tasks t5 is the default so we don't want a
    # prefix, thus the strange empty string as the key.
    "": T5_FEATURES,
}

if mt5_tasks is not None:
  MT5_FEATURES = mt5_tasks.DEFAULT_OUTPUT_FEATURES
  MODEL_TO_FEATURES["mt5_"] = MT5_FEATURES
if byt5_tasks is not None:
  BYT5_FEATURES = byt5_tasks.DEFAULT_BYTE_OUTPUT_FEATURES
  MODEL_TO_FEATURES["byt5_"] = BYT5_FEATURES

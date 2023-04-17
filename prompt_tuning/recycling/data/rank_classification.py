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

"""Utilities for creating rank classification versions of out datasets."""

import tensorflow as tf


def get_inputs(ex, labels):
  """Get inputs for each target during rank classification."""
  return [ex["inputs"]] * len(labels)


def get_targets(ex, labels):
  """Get possible targets for rank classification."""
  del ex
  return list(labels)


def get_correct(ex, labels):
  """Get a boolean mask denoting which target is correct."""
  return tf.equal(list(labels), ex["targets"])

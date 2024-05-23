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

"""Preprocessors for Prompt Recycling experiments."""

import math
from typing import Optional, Sequence
import seqio
from t5.data import preprocessors as t5_preprocessors
import tensorflow as tf


def random_targets(ds,
                   output_features,
                   sequence_length,
                   targets: Sequence[str],
                   probs: Optional[Sequence[float]] = None,
                   seed: Optional[float] = None):
  """Pick a random target from `targets` according to `probs`."""
  del sequence_length
  # Default to uniform prob over targets
  if probs is None:
    probs = [1 / len(targets)] * len(targets)
  # Allow them to not specify the final prob
  if len(probs) == len(targets) - 1:
    probs = list(probs)
    probs.append(1 - sum(probs))
  # Make sure it is a valid prob dist that covers all targets.
  if len(probs) != len(targets):
    raise ValueError("Probs not provided for all targets")
  # Check of a value greater than 1 or less than 0 because if they didn't give
  # the last probability, the code above might generate a negative or large
  # value to. Also they might have given a bad value.
  if any(p > 1.0 for p in probs):
    raise ValueError("Probability for some class is greater than one!")
  if any(p < 0.0 for p in probs):
    raise ValueError("Probability for some class is less than zero!")
  if any(p == 0.0 for p in probs):
    raise ValueError("Some class has a probability of zero, just remove this "
                     "as an option instead of setting the probability to zero.")
  if not math.isclose(math.fsum(probs), 1.0):
    raise ValueError("Probs does not total to 1.")

  # Add a fake batch dimension (once) as stateless_categorical expects it.
  probs = tf.expand_dims(probs, axis=0)
  # stateless_categorical expects log probs. We don't allow class probabilities
  # of zero so this is safe.
  probs = tf.math.log(probs)

  # Convert targets into tensors
  vocab = output_features["targets"].vocabulary
  targets = [vocab.encode_tf(t) for t in targets]
  # Get the length of each tokenized target
  target_lengths = [tf.shape(t)[0] for t in targets]
  # Convert targets to a ragged tensor so we can index it with a tensor
  targets = tf.RaggedTensor.from_row_lengths(tf.concat(targets, axis=0),
                                             target_lengths)

  @seqio.map_over_dataset(num_seeds=1)
  def _random_targets(ex, seed):
    # Select a target weighed by probs.
    idx = tf.random.stateless_categorical(probs, 1, seed)
    ex["targets"] = targets[tf.squeeze(idx)]
    return ex

  with seqio.utils.map_seed_manager(initial_seed=seed):
    # SeqIO injects the seed into this call via the context manager so this is
    # a pylint false positive.
    return _random_targets(ds)  # pylint: disable=no-value-for-parameter


@seqio.map_over_dataset
def label_idx_to_string(
    ex,
    label_names: Sequence[str],
    key: str = "label",
    unk_value: int = -1,
    unk_string: str = "UNK"):
  """Convert index based labels (0, 1, etc.) to a string and handle UNK."""
  ex[key] = tf.cond(
      tf.equal(ex[key], unk_value),
      lambda: tf.constant(unk_string),
      lambda: tf.gather(list(label_names), ex[key])
  )
  return ex


@seqio.map_over_dataset
def join_fields(
    ex,
    fields: Sequence[str] = ("title", "description"),
    output_field: str = "text"
):
  ex[output_field] = t5_preprocessors._string_join(  # pylint: disable=protected-access
      [s for f in fields if (s := ex.get(f)) is not None])
  return ex

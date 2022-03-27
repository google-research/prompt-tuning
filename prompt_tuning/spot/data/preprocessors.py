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

"""Preprocessors for SPoT tasks."""

import seqio
import t5.data.preprocessors
import tensorflow.compat.v2 as tf

# pylint:disable=no-value-for-parameter, protected-access
AUTOTUNE = tf.data.experimental.AUTOTUNE

_string_join = t5.data.preprocessors._string_join
_pad_punctuation = t5.data.preprocessors._pad_punctuation


@seqio.map_over_dataset
def preprocess_text_classification(example,
                                   text_a_key,
                                   text_b_key=None,
                                   task_name=None,
                                   label_names=None):
  """Convert a text classification dataset to a text-to-text format.

  Each {<source_text>, <target_text>} example will have the format:
  {'inputs': <task_name> <text_a_key>: <text_a> [<text_b_key>: <text_b>],
   'targets': <target_text>}

  Args:
    example: An example to process.
    text_a_key: The key for the (first) text.
    text_b_key: The key for the second text (if any).
    task_name: The name of the task.
    label_names: A list of label names corresponding to class index.

  Returns:
    A preprocessed example with the format listed above.
  """

  text_a = example[text_a_key]
  text_b = example[text_b_key] if text_b_key is not None else None

  strs_to_join = [task_name, f'{text_a_key}:', text_a, f'{text_b_key}:', text_b]

  if label_names is not None:
    label = example['label']
    if label.dtype == tf.string:
      label = tf.strings.to_number(label, tf.int64)
    label_name = tf.cond(
        # When no label is provided (label == -1), use "<unk>"
        tf.equal(label, -1),
        lambda: tf.constant('<unk>'),
        # Otherwise grab the label text from label_names
        lambda: tf.gather(label_names, label),
    )
  else:
    label_name = tf.as_string(example['label'])

  return {
      'inputs': _string_join([s for s in strs_to_join if s is not None]),
      'targets': label_name
  }

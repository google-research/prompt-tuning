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


@seqio.map_over_dataset
def preprocess_text_similarity(example,
                               text_a_key,
                               text_b_key,
                               task_name=None,
                               id_key='idx'):
  """Convert a text similarity dataset to a text-to-text format.

  This is similar to the stsb(x) preprocessor for T5 except that it allows a
  task name to be prepended to each example. Here, we consider a text similarity
  task that maps two sentences to a floating point number between 0 and 5
  representing their semantic similarity. Since we are treating all tasks as
  text-to-text tasks we need to convert this floating point number to a string.
  We first round the number to the closest entry in the set [0, 0.2, 0.4, ...,
  4.8, 5.0], and then we convert the result to a string (literally e.g. "3.4").
  This converts the dataset roughly into a 26-class classification dataset. This
  function uses the feature names from the dataset to unpack examples into a
  format amenable for a text-to-text problem.

  For example, a typical example from the CxC dataset might look like
  {
    "sentence1": "a bowl filled with oranges that are covered in water",
    "sentence2": "There is a glass bowl with oranges in it.",
    "label": 3.4,
  }

  This example would be transformed to
  {
     "inputs": (
         "cxc sentence1: a bowl filled with oranges that are covered in water "
         "sentence2: There is a glass bowl with oranges in it."
     ),
     "targets": "3.4",
  }

  Args:
    example: An example to process.
    text_a_key: The key for the first text.
    text_b_key: The key for the second text.
    task_name: The name of the task.
    id_key: str, key for id in the dataset. If not provided, 'idx' will be used.
      if None, no id will be added to the dataset.
  Returns:
    A preprocessed example.
  """
  strs_to_join = [
      task_name, f'{text_a_key}:', example[text_a_key], f'{text_b_key}:',
      example[text_b_key]
  ]
  label = tf.strings.to_number(example['label'], tf.float32)
  label_string = tf.as_string(tf.round(label * 5) / 5, precision=1)
  joined = _string_join(strs_to_join)
  return {'inputs': joined, 'targets': label_string, 'idx': example[id_key]}


@seqio.map_over_dataset
def preprocess_goemotions(example):
  """Converts the Goemotions dataset into a text-to-text format."""
  strs_to_join = []
  for feature in example.keys():
    if feature == 'comment_text':
      continue
    elif example[feature]:
      strs_to_join.append(feature)

  return {
      'inputs':
          _string_join(['goemotions', 'comment:', example['comment_text']]),
      'targets':
          tf.strings.join(strs_to_join, separator=' \\\\n '),
  }


@seqio.map_over_dataset
def preprocess_sentiment140(example, task_name=None, label_names=None):
  """Converts the Sentiment140 dataset into a text-to-text format."""
  strs_to_join = ['text:', example['text']]
  if task_name is not None:
    strs_to_join = [task_name] + strs_to_join

  if label_names is not None:
    label_name = tf.cond(
        # When no label is provided (label == -1), use "<unk>"
        tf.equal(example['polarity'], -1),
        lambda: tf.constant('<unk>'),
        # Otherwise grab the label text from label_names
        lambda: tf.gather(label_names, example['polarity'] // 2),
    )
  else:
    label_name = tf.as_string(example['polarity'])

  return {
      'inputs': _string_join(strs_to_join),
      'targets': label_name
  }


@seqio.map_over_dataset
def preprocess_gem_sgd(example, task_name=None):
  """Convert the GEM SGD dataset to a text-to-text format."""
  strs_to_join = ['prompt:', example['prompt'], 'context:', ' ']
  if task_name is not None:
    strs_to_join = [task_name] + strs_to_join
  inputs = _string_join(strs_to_join) + tf.strings.reduce_join(
      [example['context']], separator=' \\\\n ')
  return {
      'inputs': inputs,
      'targets': example['target'],
  }

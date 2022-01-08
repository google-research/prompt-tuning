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

"""Tests for preprocessors."""

import textwrap
import unittest.mock as mock
from prompt_tuning.data import preprocessors
from seqio import test_utils
import tensorflow.compat.v2 as tf


class PreprocessorsTest(tf.test.TestCase):

  def test_remove_first_text_token(self):
    input_strings = ["This is my first example", "The second"]
    gold_strings = [" ".join(s.split()[1:]) for s in input_strings]
    ds = tf.data.Dataset.from_tensor_slices({"inputs": input_strings})

    print(ds)

    processed_ds = preprocessors.remove_first_text_token(ds)

    for res, gold in zip(processed_ds, gold_strings):
      self.assertEqual(res["inputs"].numpy().decode("utf-8"), gold)

  def test_add_sentinel_to_beginning(self):
    vocab_size = 100
    offset = 0
    field = "targets"
    ds = tf.data.Dataset.from_tensor_slices({
        field: tf.zeros([3, 4], dtype=tf.int32),
    })
    output_features = {
        field: mock.MagicMock(vocabulary=mock.MagicMock(vocab_size=vocab_size))
    }

    processed_ds = preprocessors.add_sentinel_to_beginning(
        ds, output_features, field, offset)

    for ex in processed_ds:
      self.assertEqual(ex[field][0].numpy().item(), vocab_size - (offset + 1))

  def test_tsv_to_squad(self):
    fake_data = textwrap.dedent("""
    id\tcontext\tquestion\tanswer\tanswers
    0\tThe capital of France is Paris\tWhat is the capital of France?\tParis\tParis|||paris
    1\tAn ant can carry many times it's body weight making it v strong.\tAre ants strong?\tYes\tYes
    """.strip("\n"))
    ds = tf.data.Dataset.from_tensor_slices(fake_data.split("\n")[1:-1])
    ds = preprocessors.preprocess_tsv_to_qa(ds)

    gold_data = [{
        "id": "0",
        "question": "What is the capital of France ? ",
        "answer": "Paris",
        "answers": ["Paris", "paris"],
        "context": "The capital of France is Paris",
        "inputs":
            "question: What is the capital of France ? context: The capital of"
            " France is Paris",
        "targets": "Paris"
    }, {
        "id":
            "1",
        "question":
            "Are ants strong ? ",
        "answer":
            "Yes",
        "answers": ["Yes"],
        "context":
            "An ant can carry many times it ' s body weight making it v strong . ",
        "inputs":
            "question: Are ants strong ? context: An ant can carry many times "
            "it ' s body weight making it v strong . ",
        "targets":
            "Yes"
    }]

    for ex, gold in zip(ds, gold_data):
      self.assertEqual(ex["id"].numpy().decode("utf-8"), gold["id"])
      self.assertEqual(ex["question"].numpy().decode("utf-8"), gold["question"])
      self.assertEqual(ex["answer"].numpy().decode("utf-8"), gold["answer"])
      self.assertEqual(ex["context"].numpy().decode("utf-8"), gold["context"])
      self.assertEqual(ex["targets"].numpy().decode("utf-8"), gold["targets"])
      for answer, gold_answer in zip(ex["answers"].numpy(), gold["answers"]):
        self.assertEqual(answer.decode("utf-8"), gold_answer)

  def test_preprocess_text_generation(self):
    example = tf.data.Dataset.from_tensor_slices({
        "source_aligned": {
            "en": ["english input"],
            "es": ["spanish input"]
        },
        "target_aligned": {
            "en": ["english target"],
            "es": ["spanish target"]
        }
    })
    processed_example = preprocessors.preprocess_text_generation(
        example,
        source_key="source_aligned",
        target_key="target_aligned",
        task_name=None,
        prefix="summarize:",
        source_nested_key="en",
        target_nested_key="es",
    )
    test_utils.assert_dataset(processed_example, {
        "inputs": "summarize: english input",
        "targets": "english target"
    })


if __name__ == "__main__":
  tf.test.main()

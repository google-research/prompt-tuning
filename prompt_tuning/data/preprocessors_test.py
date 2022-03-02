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

import os
import textwrap
import unittest.mock as mock
from absl.testing import parameterized
import numpy as np
from prompt_tuning.data import preprocessors
import seqio
from seqio import test_utils
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


TEST_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data")


INPUTS_SIZE = 10
TARGETS_SIZE = 5
TEXT_SIZE = 10


TEST_T5_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=seqio.SentencePieceVocabulary(
            os.path.join(TEST_DATA, "t5_vocab"), 100),
        add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=seqio.SentencePieceVocabulary(
            os.path.join(TEST_DATA, "t5_vocab"), 100),
        add_eos=True)
}


def create_fake_text_dataset(examples: int = 10, text_size: int = TEXT_SIZE):
  text = np.reshape(
      # Start at 2 so we skip EOS=1 which could be a problem on any tests that
      # actually decode the fake inputs.
      np.arange(2, examples * text_size + 2),
      (-1, text_size)).astype(np.int32)
  return tf.data.Dataset.from_tensor_slices({"targets": text})


class PreprocessorsTest(tf.test.TestCase):

  def test_remove_first_text_token(self):
    input_strings = ["This is my first example", "The second"]
    gold_strings = [" ".join(s.split()[1:]) for s in input_strings]
    ds = tf.data.Dataset.from_tensor_slices({"inputs": input_strings})

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

  def test_tsv_to_qa(self):
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
        "targets": "spanish target"
    })


class BARTTaskTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="text_infilling",
           preprocessor=preprocessors.text_infilling),
      dict(testcase_name="token_deletion",
           preprocessor=preprocessors.token_deletion))
  def test_inputs_shorter_than_targets(self, preprocessor):
    ds = create_fake_text_dataset()
    ds = preprocessor(ds,
                      {"inputs": INPUTS_SIZE + 1, "targets": TARGETS_SIZE + 1},
                      TEST_T5_FEATURES,
                      noise_density=0.5)
    for ex in tfds.as_numpy(ds):
      self.assertLess(ex["inputs"].shape[0], ex["targets"].shape[0])

  @parameterized.named_parameters(
      dict(testcase_name="text_infilling",
           preprocessor=preprocessors.text_infilling),
      dict(testcase_name="token_deletion",
           preprocessor=preprocessors.token_deletion))
  def test_extra_id_not_in_targets(self, preprocessor):
    ds = create_fake_text_dataset()
    ds = preprocessor(ds,
                      {"inputs": INPUTS_SIZE + 1, "targets": TARGETS_SIZE + 1},
                      TEST_T5_FEATURES,
                      noise_density=0.5)
    vocab = TEST_T5_FEATURES["targets"].vocabulary
    for ex in tfds.as_numpy(ds):
      targets_text = vocab.decode(ex["targets"].tolist())
      self.assertNotIn("extra_id", targets_text)

  @parameterized.named_parameters(
      dict(testcase_name="text_infilling",
           preprocessor=preprocessors.text_infilling),
      dict(testcase_name="token_deletion",
           preprocessor=preprocessors.token_deletion))
  def test_target_tokens_match_original_tokens(self, preprocessor):
    ds = create_fake_text_dataset()
    processed_ds = preprocessor(
        ds,
        {"inputs": INPUTS_SIZE + 1, "targets": TARGETS_SIZE + 1},
        TEST_T5_FEATURES,
        noise_density=0.5)
    for processed_ex, ex in zip(tfds.as_numpy(processed_ds), tfds.as_numpy(ds)):
      np.testing.assert_array_equal(processed_ex["targets"], ex["targets"])

  def test_extra_id_not_in_token_deletion_inputs(self):
    ds = create_fake_text_dataset()
    ds = preprocessors.token_deletion(
        ds,
        {"inputs": INPUTS_SIZE + 1, "targets": TARGETS_SIZE + 1},
        TEST_T5_FEATURES,
        noise_density=0.5)
    vocab = TEST_T5_FEATURES["inputs"].vocabulary
    for ex in tfds.as_numpy(ds):
      inputs_text = vocab.decode(ex["inputs"].tolist())
      self.assertNotIn("extra_id", inputs_text)

  def test_extra_id_in_text_infilling_inputs(self):
    ds = create_fake_text_dataset()
    ds = preprocessors.text_infilling(
        ds,
        {"inputs": INPUTS_SIZE + 1, "targets": TARGETS_SIZE + 1},
        TEST_T5_FEATURES,
        noise_density=0.5)
    vocab = TEST_T5_FEATURES["inputs"].vocabulary
    for ex in tfds.as_numpy(ds):
      inputs_text = vocab.decode(ex["inputs"].tolist())
      self.assertIn("extra_id", inputs_text)


if __name__ == "__main__":
  tf.test.main()

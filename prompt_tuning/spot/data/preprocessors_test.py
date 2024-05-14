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

"""Tests for SPoT preprocessors."""

from absl.testing import absltest
from prompt_tuning.spot.data import preprocessors as spot_preprocessors
from seqio import test_utils
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

assert_dataset = test_utils.assert_dataset


class PreprocessorsTest(tf.test.TestCase):

  def test_preprocess_text_classification_single_text(self):
    builder_config = tfds.text.glue.Glue.builder_configs['sst2']
    original_dataset = tf.data.Dataset.from_tensor_slices({
        'sentence': [
            'for the uninitiated plays better on video with the sound',
            'like a giant commercial for universal studios , where much of the action takes place',
            'company once again dazzle and delight us',
            "'s no surprise that as a director washington demands and receives excellent performances , from himself and from newcomer derek luke",
            ', this cross-cultural soap opera is painfully formulaic and stilted .'
        ],
        'label': [0, 0, 1, 1, 0],
    })
    dataset = spot_preprocessors.preprocess_text_classification(
        original_dataset,
        text_a_key='sentence',
        task_name='sst2',
        label_names=builder_config.label_classes)
    assert_dataset(dataset, [
        {
            'inputs':
                'sst2 sentence: for the uninitiated plays better on video with the sound',
            'targets':
                'negative',
        },
        {
            'inputs':
                'sst2 sentence: like a giant commercial for universal studios , where much of the action takes place',
            'targets':
                'negative',
        },
        {
            'inputs':
                'sst2 sentence: company once again dazzle and delight us',
            'targets':
                'positive',
        },
        {
            'inputs':
                "sst2 sentence: 's no surprise that as a director washington demands and receives excellent performances , from himself and from newcomer derek luke",
            'targets':
                'positive',
        },
        {
            'inputs':
                'sst2 sentence: , this cross-cultural soap opera is painfully formulaic and stilted .',
            'targets':
                'negative',
        },
    ])

  def test_preprocess_text_classification_text_pair(self):
    builder_config = tfds.text.glue.Glue.builder_configs['mnli']
    original_dataset = tf.data.Dataset.from_tensor_slices({
        'hypothesis': [
            'The Clinton followers kept to the higher ground in the discussion.',
            'Women have jobs in all areas of the workforce, they are almost getting the same wages as most men,',
            'Houston is freezing and dry right now.',
            "But they wouldn't be leaving right now.",
            'Ask at CiteHall center about pet supplies.',
        ],
        'premise': [
            'The Clinton surrogates also held the high ground in the context war.',
            "um-hum because women are in every field now i mean i can't think of a field that they're not involved in",
            'Houston is really humid now',
            'But not now.',
            'Ask at CiteHall center about advance reservations.',
        ],
        'label': [0, 1, 2, 1, 2],
    })
    dataset = spot_preprocessors.preprocess_text_classification(
        original_dataset,
        text_a_key='hypothesis',
        text_b_key='premise',
        task_name='mnli',
        label_names=builder_config.label_classes)
    assert_dataset(dataset, [
        {
            'inputs':
                'mnli hypothesis: The Clinton followers kept to the higher ground in the discussion. premise: The Clinton surrogates also held the high ground in the context war.',
            'targets':
                'entailment',
        },
        {
            'inputs':
                "mnli hypothesis: Women have jobs in all areas of the workforce, they are almost getting the same wages as most men, premise: um-hum because women are in every field now i mean i can't think of a field that they're not involved in",
            'targets':
                'neutral',
        },
        {
            'inputs':
                'mnli hypothesis: Houston is freezing and dry right now. premise: Houston is really humid now',
            'targets':
                'contradiction',
        },
        {
            'inputs':
                "mnli hypothesis: But they wouldn't be leaving right now. premise: But not now.",
            'targets':
                'neutral',
        },
        {
            'inputs':
                'mnli hypothesis: Ask at CiteHall center about pet supplies. premise: Ask at CiteHall center about advance reservations.',
            'targets':
                'contradiction',
        },
    ])


if __name__ == '__main__':
  absltest.main()

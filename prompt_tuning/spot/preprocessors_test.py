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

"""Tests for SPoT preprocessors."""

from absl.testing import absltest
from prompt_tuning.data import preprocessors as pt_preprocessors
from prompt_tuning.spot import preprocessors as spot_preprocessors
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
            'inputs': 'sst2 sentence: company once again dazzle and delight us',
            'targets': 'positive',
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

  def test_preprocess_text_similarity(self):
    original_dataset = tf.data.Dataset.from_tensor_slices({
        'sentence1': [
            "A young girl is sitting on Santa's lap.",
            'A women sitting at a table drinking with a basketball picture in the background.',
            'Norway marks anniversary of massacre',
            'US drone kills six militants in Pakistan: officials',
            'On Tuesday, the central bank left interest rates steady, as expected, but also declared that overall risks were weighted toward weakness and warned of deflation risks.',
        ],
        'sentence2': [
            "A little girl is sitting on Santa's lap",
            'A woman in a sari drinks something while sitting at a table.',
            "Norway Marks Anniversary of Breivik's Massacre",
            'US missiles kill 15 in Pakistan: officials',
            "The central bank's policy board left rates steady for now, as widely expected, but surprised the market by declaring that overall risks were weighted toward weakness.",
        ],
        'label': [
            4.800000190734863, 2.5999999046325684, 3.799999952316284,
            2.4000000953674316, 4.0
        ],
        'idx': [1680, 1456, 4223, 4093, 3017],
    })
    dataset = spot_preprocessors.preprocess_text_similarity(
        original_dataset,
        text_a_key='sentence1',
        text_b_key='sentence2',
        task_name='stsb',
        id_key='idx',
    )
    assert_dataset(dataset, [
        {
            'inputs':
                "stsb sentence1: A young girl is sitting on Santa's lap. sentence2: A little girl is sitting on Santa's lap",
            'targets':
                '4.8',
            'idx':
                1680,
        },
        {
            'inputs':
                "mnli hypothesis: Women have jobs in all areas of the workforce, they are almost getting the same wages as most men, premise: um-hum because women are in every field now i mean i can't think of a field that they're not involved in",
            'targets':
                '2.6',
            'idx':
                1456,
        },
        {
            'inputs':
                'mnli hypothesis: Houston is freezing and dry right now. premise: Houston is really humid now',
            'targets':
                '3.8',
            'idx':
                4223,
        },
        {
            'inputs':
                "mnli hypothesis: But they wouldn't be leaving right now. premise: But not now.",
            'targets':
                '2.4',
            'idx':
                4093,
        },
        {
            'inputs':
                'mnli hypothesis: Ask at CiteHall center about pet supplies. premise: Ask at CiteHall center about advance reservations.',
            'targets':
                '4.0',
            'idx':
                3017,
        },
    ])

  def test_preprocess_goemotions(self):
    original_dataset = tf.data.Dataset.from_tensor_slices({
        'admiration': [
            False,
            True,
            False,
            False,
            False,
        ],
        'amusement': [
            False,
            False,
            False,
            False,
            False,
        ],
        'anger': [
            False,
            False,
            False,
            False,
            False,
        ],
        'annoyance': [
            False,
            False,
            False,
            False,
            False,
        ],
        'approval': [
            False,
            False,
            False,
            False,
            False,
        ],
        'caring': [
            False,
            False,
            False,
            False,
            False,
        ],
        'comment_text': [
            "It's just wholesome content, from questionable sources",
            'This is actually awesome.',
            "People really spend more than $10 in an app game? I mean an actual video game I can understand but that's just...sad",
            'I grew up on the other side of Ama but live in Tulia now. I will have some El Burrito for you',
            'What the problem? I mean, steak? Good. Doughnuts? Good!! I don’t see an issue.',
        ],
        'confusion': [
            False,
            False,
            True,
            False,
            False,
        ],
        'curiosity': [
            False,
            False,
            False,
            False,
            True,
        ],
        'desire': [
            False,
            False,
            False,
            False,
            False,
        ],
        'disappointment': [
            False,
            False,
            False,
            False,
            False,
        ],
        'disapproval': [
            False,
            False,
            False,
            False,
            True,
        ],
        'disgust': [
            False,
            False,
            False,
            False,
            False,
        ],
        'embarrassment': [
            False,
            False,
            False,
            False,
            False,
        ],
        'excitement': [
            False,
            False,
            False,
            False,
            False,
        ],
        'fear': [
            False,
            False,
            False,
            False,
            False,
        ],
        'gratitude': [
            False,
            False,
            False,
            False,
            False,
        ],
        'grief': [
            False,
            False,
            False,
            False,
            False,
        ],
        'joy': [
            False,
            False,
            False,
            False,
            False,
        ],
        'love': [
            False,
            False,
            False,
            False,
            False,
        ],
        'nervousness': [
            False,
            False,
            False,
            False,
            False,
        ],
        'neutral': [
            True,
            False,
            False,
            True,
            False,
        ],
        'optimism': [
            False,
            False,
            False,
            False,
            False,
        ],
        'pride': [
            False,
            False,
            False,
            False,
            False,
        ],
        'realization': [
            False,
            False,
            False,
            False,
            False,
        ],
        'relief': [
            False,
            False,
            False,
            False,
            False,
        ],
        'remorse': [
            False,
            False,
            False,
            False,
            False,
        ],
        'sadness': [
            False,
            False,
            True,
            False,
            False,
        ],
        'surprise': [
            False,
            False,
            False,
            False,
            False,
        ],
    })
    dataset = spot_preprocessors.preprocess_goemotions(original_dataset)
    assert_dataset(dataset, [
        {
            'inputs':
                "goemotions comment: It's just wholesome content, from questionable sources",
            'targets':
                'neutral',
        },
        {
            'inputs': 'goemotions comment: This is actually awesome.',
            'targets': 'admiration',
        },
        {
            'inputs':
                "goemotions comment: People really spend more than $10 in an app game? I mean an actual video game I can understand but that's just...sad",
            'targets':
                'confusion \\n sadness',
        },
        {
            'inputs':
                'goemotions comment: I grew up on the other side of Ama but live in Tulia now. I will have some El Burrito for you',
            'targets':
                'neutral',
        },
        {
            'inputs':
                'goemotions comment: What the problem? I mean, steak? Good. Doughnuts? Good!! I don’t see an issue.',
            'targets':
                'curiosity \\n disapproval',
        },
    ])

  def test_preprocess_sentiment140(self):
    original_dataset = tf.data.Dataset.from_tensor_slices({
        'text': [
            "i'm 10x cooler than all of you!",
            'O.kk? Thats weird I cant stop following people on twitter... I have tons of people to unfollow',
            'what a beautiful day not to got to my first class',
            ".@HildyGottlieb &amp; I was just saying to Maha'al yesterday, everything we ever needed to know was in Beatles' lyrics - you prove my point!",
            'kinda sad and confused why do guys do this?',
        ],
        'polarity': [4, 0, 4, 4, 0],
    })
    dataset = spot_preprocessors.preprocess_sentiment140(
        original_dataset,
        task_name='sentiment140',
        label_names=['negative', 'neutral', 'positive']),
    assert_dataset(dataset, [
        {
            'inputs': "sentiment140 text: i'm 10x cooler than all of you!",
            'targets': 'positive',
        },
        {
            'inputs':
                'sentiment140 text: O.kk? Thats weird I cant stop following people on twitter... I have tons of people to unfollow',
            'targets':
                'negative',
        },
        {
            'inputs':
                'sentiment140 text: what a beautiful day not to got to my first class',
            'targets':
                'positive',
        },
        {
            'inputs':
                "sentiment140 text: .@HildyGottlieb &amp; I was just saying to Maha'al yesterday, everything we ever needed to know was in Beatles' lyrics - you prove my point!",
            'targets':
                'positive',
        },
        {
            'inputs':
                'sentiment140 text: kinda sad and confused why do guys do this?',
            'targets':
                'negative',
        },
    ])

  def test_preprocess_text_generation_gem_common_gen(self):
    original_dataset = tfds.load(
        'gem/common_gen:1.1.0', split='validation', shuffle_files=False).take(5)
    dataset = pt_preprocessors.preprocess_text_generation(
        original_dataset,
        source_key='concepts',
        target_key='target',
        task_name='common_gen',
        source_segment=' \\\\n ',
    )
    assert_dataset(dataset, [
        {
            'inputs':
                'common_gen hand \\n hold \\n knife \\n orange \\n peel',
            'targets':
                'Steadily holding the knife in hand, Sarah peeled the rind from the orange.',
        },
        {
            'inputs':
                'common_gen face \\n foot \\n laugh \\n say \\n stick',
            'targets':
                'The girl says that faces start laughing as the feet stick together and fall.',
        },
        {
            'inputs': 'common_gen catch \\n fish \\n river',
            'targets': 'The girl will go to the river and try to catch a fish.',
        },
        {
            'inputs':
                'common_gen instrument \\n march \\n street',
            'targets':
                'The band marched down the street with many instruments.',
        },
        {
            'inputs':
                'common_gen house \\n look \\n walk',
            'targets':
                'While the men were walking, they looked at the house across the street.',
        },
    ])

  def test_preprocess_text_generation_gem_dart(self):
    original_dataset = tfds.load(
        'gem/dart:1.1.0', split='validation', shuffle_files=False).take(5)
    dataset = pt_preprocessors.preprocess_text_generation(
        original_dataset,
        source_key='tripleset',
        target_key='target',
        task_name='dart',
        source_segment=' \\\\n ',
    )
    assert_dataset(dataset, [
        {
            'inputs':
                'dart The Golden Palace eatType coffee shop \\n The Golden Palace food Chinese \\n The Golden Palace priceRange moderate \\n The Golden Palace customer rating 1 out of 5 \\n The Golden Palace area city centre',
            'targets':
                'The Golden Palace is a coffee shop that provides Chinese food at moderate prices with a 1 out of 5 customer rating in city centre.',
        },
        {
            'inputs':
                ' dartThe Golden Palace eatType coffee shop \\n The Golden Palace food Chinese \\n The Golden Palace priceRange moderate \\n The Golden Palace customer rating average \\n The Golden Palace area riverside',
            'targets':
                'In the riverside area is the coffee shop The Golden Palace.  It offers Chinese food in the higher price range and has an average customer rating.',
        },
        {
            'inputs':
                'dart 3Arena OWNER Live Nation Entertainment \\n Live Nation Entertainment LOCATION Beverly Hills, California',
            'targets':
                '3Arena, owned by Live Nation Entertainment, is located in Beverly Hills, California.',
        },
        {
            'inputs':
                'dart Cotto eatType restaurant \\n Cotto food Chinese \\n Cotto priceRange moderate \\n Cotto customer rating 1 out of 5 \\n Cotto area riverside \\n Cotto near The Portland Arms',
            'targets':
                'Cotto is a coffee shop that serves moderate priced Chinese food.  This restaurant has a customer rating of 1 out of 5 and is located in riverside near The Portland Arms.',
        },
        {
            'inputs':
                'dart Al Asad Airbase OPERATING_ORGANISATION United States Air Force \\n United States Air Force BATTLES United States invasion of Panama',
            'targets':
                'The United States invasion of Panama was a battle involving the United States Air Force who operate the Al Asad Airbase.',
        },
    ])

  def test_preprocess_text_generation_gem_web_nlg_en(self):
    original_dataset = tfds.load(
        'gem/web_nlg_en:1.1.0', split='validation', shuffle_files=False).take(5)
    dataset = pt_preprocessors.preprocess_text_generation(
        original_dataset,
        source_key='input',
        target_key='target',
        task_name='web_nlg_en',
        source_segment=' \\\\n ',
    )
    assert_dataset(dataset, [
        {
            'inputs':
                'web_nlg_en Auron_(comicsCharacter) | creator | Karl_Kesel \\n Karl_Kesel | nationality | Americans \\n Auron_(comicsCharacter) | creator | Walt_Simonson',
            'targets':
                'The comic character Auron was created by Walt Simonson and the American, Karl Kesel.',
        },
        {
            'inputs':
                'web_nlg_en Above_the_Veil | country | Australians \\n Into_Battle_(novel) | followedBy | The_Violet_Keystone \\n Above_the_Veil | followedBy | Into_Battle_(novel) \\n Above_the_Veil | precededBy | Aenir \\n Aenir | precededBy | Castle_(novel)',
            'targets':
                'Above the Veil is an Australian novel and the sequel to Aenir and Castle. It was later followed by Into Battle and The Violet Keystone.',
        },
        {
            'inputs':
                'web_nlg_en Akeem_Dent | debutTeam | Atlanta_Falcons \\n Akeem_Dent | birthDate | 1987-09-27 \\n Akeem_Dent | birthPlace | "Atlanta, Georgia"',
            'targets':
                'Akeem Dent, who made his debut with the Atlanta Falcons, was born in Atlanta, Georgia on 27 September 1987.',
        },
        {
            'inputs':
                'web_nlg_en Buzz_Aldrin | mission | Apollo_11 \\n Apollo_11 | operator | NASA',
            'targets':
                'The Apollo 11 program was organized by NASA and included Buzz Aldrin as one of its crew members.',
        },
        {
            'inputs':
                'web_nlg_en A_Wizard_of_Mars | mediaType | Hardcover \\n A_Wizard_of_Mars | author | Diane_Duane \\n A_Wizard_of_Mars | isbnNumber | "978-0-15-204770-2"',
            'targets':
                'Diane Duane wrote A Wizard of Mars which is published in hardcover and has the ISBN number 978-0-15-204770-2.',
        },
    ])


if __name__ == '__main__':
  absltest.main()

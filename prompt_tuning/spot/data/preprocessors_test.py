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

  def test_preprocess_gem_schema_guided_dialog(self):
    original_dataset = tfds.load(
        'gem/schema_guided_dialog:1.1.0',
        split='validation',
        shuffle_files=False).take(5)
    dataset = spot_preprocessors.schema_guided_dialog(
        original_dataset,
        task_name='schema_guided_dialog',
    )
    assert_dataset(dataset, [
        {
            'inputs':
                'schema_guided_dialog prompt: This looks perfect for me. Please make a reservation at this hotel. context: I need to book a room in nice hotel, please search the best one for me. \\n Do you have any particular City to search for? \\n Yes, Philly is the City I need to search for the hotel. \\n Got 10 results and Aka University City is the nice one which is a 3 star hotel, how about your opinion? \\n Is there any other hotel that suits my requirement? \\n Yes, Aloft Philadelphia airport hotel with 3 star. Is that okay to you? \\n No, please search for something else. \\n What about Aloft PHiladelphia Downtown hotel with 3 star? \\n Let me know the cost and address of this hotel before to confirm. \\n Cost for 1 room per night is $120 and the address of this hotel is 101 North Broad Street.',
            'targets':
                'From which date do you want me to book this hotel and how many days you want to stay there?',
        },
        {
            'inputs':
                "schema_guided_dialog prompt: That will work. context: I'm looking for something to do that will arouse my curiosity. \\n Where should I look? What event type do you prefer? \\n A Sports event in LA would be great. \\n One of 10 events I found is on March 12th at 8:30 pm. Dodger Stadium is hosting the Dodgers Vs Cubs game which many people like. \\n Sounds fun. What else did you find in the event category? \\n Another one many people enjoy is the Dodgers Vs Diamondbacks. Dodger stadium is hosting this event at 7 pm on March 12th. \\n I really like that one. I need a bus from SF going there on the 11th of March. \\n I found a bus with 0 transfers leaving at 7 am for $43.",
            'targets':
                'Shall I secure the bus tickets by purchasing them?',
        },
        {
            'inputs':
                'schema_guided_dialog prompt: Yes, I am interested to visit there. context: Hi, i need a help, i am planing to change my apartment, can you find a good apartment for me? \\n Sure, I will help you, Which city should i search in for property? how many bed room do you need? \\n It should have three bed rooms and i need apartment in Santa Rosa \\n I found 10 apartments for you, One of the best apartment is Behringer Harvard located at 4656 Quigg drive. the apartment has 3 bed room and 3 baths. Monthly rent for the apartment is $4,050 \\n Ok, Find another apartment for me. \\n I found another apartment for you, One of the best apartment is Bicentennial Apartments located at 825 Russell Avenue. The apartment has 3 bed room and 2 baths. Monthly rent for the apartment is $3,700 \\n Ok, Please Find another apartments for me \\n I found another good apartment for you, Brush Creek Apartments is a One of the best apartment located at 6221 Montecito boulevard. The apartment has 3 bed room and 2 baths. Monthly rent for the apartment is $3,950 \\n Ok,that one suitable for me \\n Are you interested to schedule a visit to the property?',
            'targets':
                'Which date you need to schedule your visit?',
        },
        {
            'inputs':
                'schema_guided_dialog prompt: It will be for 4:15 in the evening and please name it Cleaning. context: Can you please show me the current alarms I have set? \\n You have 2 alarms set, the first is for 7 am that is named Wake Up. \\n Are there any others? \\n There is another alarm at 9 am with a name Shuttle to work. \\n Okay, thank you. I would like to add an alarm. \\n What time should I set the alarm for.',
            'targets':
                'Please confirm the following: Creating an alarm for 4:15 pm with the name Cleaning.',
        },
        {
            'inputs':
                'schema_guided_dialog prompt: Ok. I need to check my savings account balance. context: I need to find a one way flight. \\n Wheen do you want to travel and where too? \\n I want to go to Seattle later today. \\n What city will you depart from? \\n I will fly from Mexico City. \\n Do you want to fly with American Airlines? They have a flight that takes off at 11:40 am local time with 1 layover. The ticket price is $298 \\n No, find me something else. I have 0 bags to check in. \\n Do you want to fly with Delta Airlines. They have a flight that takes off at 6:35 pm local time with 1 layover. The ticket price is $315. \\n Is this for one passenger? What is the destination arrival time? \\n This is for 1 passenger. The flight will arrive at 1:17 am local time.',
            'targets':
                'Your savings account balance is $10,041.51.',
        },
    ])

  def test_preprocess_multiple_choice_cosmo_qa(self):
    original_dataset = tfds.load(
        'cosmos_qa:1.0.0', split='validation', shuffle_files=False).take(5)
    dataset = spot_preprocessors.preprocess_multiple_choice(
        original_dataset,
        choice_keys=['answer0', 'answer1', 'answer2', 'answer3'],
        context_key='context',
        question_key='question',
        task_name='cosmo_qa',
    )
    assert_dataset(dataset, [{
        'inputs':
            "cosmo_qa question: Why do you think the days preceding today have dragged for you ? answer0: None of the above choices . answer1: I am excited about abnd camp and moving and whenever excited the days leading up to an exciting event take forever to go by answer2: I did n't do anything to save up money for the band camp and move so I spent my days at home and that made time take long answer3: I had to work overtime a lot over those days and that made time go slowly context: I ca n't believe it 's already the 9th of August ! This is so exciting . Only six more days until I can move into my room in Wilmer - Davis , and then only two more days after that till band camp starts . I thought this last two weeks would drag on and on , and they have a little bit , but now that I ' m really and truly in the homestretch , the days are going to start flying by .",
        'targets':
            'I am excited about abnd camp and moving and whenever excited the days leading up to an exciting event take forever to go by'
    }, {
        'inputs':
            "cosmo_qa question: What may have happened if three women had not gotten on the bus ? answer0: They would have caught that bus answer1: They would n't have caught that bus answer2: None of the above choices . answer3: The bus would have been full context: At some point , I can finally see the doors - those three women are getting on the bus . Then , the doors close . The bus is full . Got ta wait for the next one . At that point , I just had to give up .",
        'targets':
            'They would have caught that bus'
    }, {
        'inputs':
            "cosmo_qa question: Why did the narrator and his office workers play paintball ? answer0: They could n't think of anything better to play together . answer1: The narrator 's family owned the paintball place so it was the perfect place to go . answer2: They decided that having group activities would help them bond . answer3: None of the above choices . context: I have found a new lover , and her name is Paintball ! ! ! ! ! It started a few weeks ago when a bunch of us from the office went out for a game as a bit of a team building exercise . From there it blossomed into me picking up my own marker ( gun ) and the rest of the gear . Last night I took my new marker out for a test run the arena near my house , was a great deal of fun . While Paris is not very thrilled with a sport where I go around and shoot people she is at least supportive of the fact I am getting out and getting some exercise .",
        'targets':
            'They decided that having group activities would help them bond .'
    }, {
        'inputs':
            "cosmo_qa question: Why did they go to a hot pot place ? answer0: None of the above choices . answer1: The wanted an experience similar to what is found in Manhattan . answer2: To get food similar to what one would get in Singapore . answer3: To feel like they were having an American meal in a strange country . context: It looked nothing like the one in Manhattan . It really felt like I was in Taiwan , Hong Kong or China , not that I ' ve been to those places , but it did n't feel like America . My friend took me to a hot pot place ( also known as steamboat in Singapore ) .",
        'targets':
            'To get food similar to what one would get in Singapore .'
    }, {
        'inputs':
            "cosmo_qa question: Why am I excited about going camping up North ? answer0: I am just happy that I will get to spend more time with her . answer1: I am excited to learn about some of the native plants . answer2: None of the above choices . answer3: I want to get more basil and sage plants for our garden . context: Luckily this coming weekend we 're going camping up north so I 'll get to spend some more time with her then before she goes . Yesterday , despite the extreme heat , I really wanted to get started on the back garden . The basil and sage plants that I had bought last weekend were doing okay but I knew they would do much better in the ground . We went to the market in the morning and I got a few more plants that were on my list , but there were n't as many herbs or plants for sale . I guess it 's pretty much the end of the season for putting in gardens .",
        'targets':
            'I am just happy that I will get to spend more time with her .'
    }])

  def test_preprocess_multiple_choice_hellaswag(self):
    original_dataset = tfds.load(
        'hellaswag:1.1.0', split='validation', shuffle_files=False).take(5)
    dataset = spot_preprocessors.preprocess_multiple_choice(
        original_dataset,
        choice_keys=['endings', 'endings', 'endings', 'endings'],
        choice_nested_keys=[0, 1, 2, 3],
        question_key='context',
        task_name='hellaswag',
    )
    assert_dataset(dataset, [{
        'inputs':
            "hellaswag context: [header] How to reverse tooth decay [title] Brush your teeth twice per day. [step] Regular daily brushing is crucial for maintaining good dental health and it may also help to reverse the decay process. Make sure that you brush your teeth at least twice per day, such as in the morning and before you go to bed. endings 0: [substeps] You should avoid using a toothpaste that contains fluoride. Toothpaste containing fluoride will increase the risk of tooth decay. endings 1: [substeps] Choose a toothpaste that contains fluoride. Fluoride is necessary for remineralizing teeth and reversing tooth decay. endings 2: This should help to prevent them from becoming overgrown. [title] Use a soft bristled toothbrush. endings 3: Stop at the halfway point of your brushing routine and don't forget to brush for a few seconds and then rinse your mouth. [substeps] Brushing is especially important during the daytime when you're traveling or if your periodontal infections or fever are prevalent.",
        'targets':
            '[substeps] Choose a toothpaste that contains fluoride. Fluoride is necessary for remineralizing teeth and reversing tooth decay.'
    }, {
        'inputs':
            "hellaswag context: [header] How to modify a prenuptial agreement [title] Go over the agreement with your spouse. [step] Review each provision in your prenuptial agreement to decide which provisions you and your spouse want to modify. [substeps] A prenuptial or postnuptial agreement is only valid after full disclosure from both you and your spouse regarding your income, assets, debts, and liabilities. endings 0: If any of this information has changed since you signed the prenup, share it openly. Don't assume that your spouse knows. endings 1: [title] Read and review the prenuptial agreement. [step] Include any clauses or provisions which you or your spouse agree to modify. endings 2: [title] Obtain a copy of the prenuptial agreement. [step] Once you've reviewed all of the provisions in the agreement, you'll want to make sure you have an exact copy of the prenuptial agreement. endings 3: With postnuptial agreements, you are legally bound by the terms you satisfy with the other spouse. [title] Draft the proposed or modified prenuptial agreement.",
        'targets':
            "If any of this information has changed since you signed the prenup, share it openly. Don't assume that your spouse knows."
    }, {
        'inputs':
            'hellaswag context: A man is standing holding a knife over a table. the man endings 0: proceeds to place wax on a piece of paper. endings 1: demonstrates other techniques for wounding a patient. endings 2: points to several parts of the knife. endings 3: then takes a sharpener and cuts two slices of lemon.',
        'targets':
            'points to several parts of the knife.'
    }, {
        'inputs':
            "hellaswag context: [header] How to detangle hair without pain [title] Check for dryness. [step] Before you attempt to take out any tangles, make sure your hair is not severely dry. This will make it less painful when you are actually taking the knots out. endings 0: Run your fingers through your hair to see if it's slightly sticky to the touch. If it is, try rubbing some old hair conditioner into it. endings 1: Run your fingers through your hair to ensure that it is not too damp and that it is looking sleek. [substeps] Normal hair can feel rough or lifeless if it is over-dry. endings 2: [title] Separate the hair completely if you have it tied up and leave it free. [step] Attempt to free any loose hairs from a tangled clump. endings 3: Wet hair tends to dry very quickly and can cause less hair damage. [substeps] If the oil in your hair helps to detangle it while you're detangling it, hold it over a sink or tub and then gently pull the roots and tips of your hair downward to remove as much oil as possible.",
        'targets':
            '[title] Separate the hair completely if you have it tied up and leave it free. [step] Attempt to free any loose hairs from a tangled clump.'
    }, {
        'inputs':
            'hellaswag context: [header] How to control cravings [title] Distract yourself from your craving. [step] Help to take your mind off of your craving by putting your attention into something else you enjoy. Your brain will be satisfied that you are doing something stimulating, and focus less on the need to fulfill your craving. endings 0: How you distract yourself will vary from person to person, so choose something you enjoy. [substeps] For example, instead of having dessert after dinner, go for a walk. endings 1: [substeps] Distraction is one of the first ways your brain has to reset. Distract yourself by typing a little distraction blog or article into your favorite browser. endings 2: This can be something like : [substeps] Reading a book or other entertainment like an interest take a walk with your dog play board games add new fun to your day [title] Make sure you get plenty of sleep the night before the craving occurs. [step] If you are exhausted by that latest movie or tv show, you may end up more emotionally hungry. endings 3: Many people find quiet relaxes a sore pair of muscles, and quiet can help them feel relaxed. [substeps] If you find yourself excessively feeling out of control during a craving, read a book instead.',
        'targets':
            'How you distract yourself will vary from person to person, so choose something you enjoy. [substeps] For example, instead of having dessert after dinner, go for a walk.'
    }])

  def test_preprocess_multiple_choice_piqa(self):
    original_dataset = tfds.load(
        'piqa:1.0.0', split='validation', shuffle_files=False).take(5)
    dataset = spot_preprocessors.preprocess_multiple_choice(
        original_dataset,
        choice_keys=['sol1', 'sol2'],
        question_key='goal',
        task_name='piqa',
    )
    assert_dataset(dataset, [
        {
            'inputs':
                "piqa goal: To pour hot fudge over ice cream before serving, sol1: pour the hot fudge over ice cream that has just been pulled from the freezer and scooped out of it's container with an ice cream scoop into a bowl. sol2: pour the hot fudge over ice cream that has been pulled out of the freezer and softened for fifteen minutes, then scooped out of it's container with an ice cream scoop into a bowl.",
            'targets':
                "pour the hot fudge over ice cream that has just been pulled from the freezer and scooped out of it's container with an ice cream scoop into a bowl."
        },
        {
            'inputs':
                'piqa goal: How can I make paper look older? sol1: Dip paper in some black tea. sol2: Dip paper in some green tea.',
            'targets':
                'Dip paper in some black tea.'
        },
        {
            'inputs':
                'piqa goal: how do you flex? sol1: show your muscles. sol2: lay down.',
            'targets':
                'show your muscles.'
        },
        {
            'inputs':
                'piqa goal: How do you clean inside windows? sol1: Spray window cleaner on glass. For the best streak-free results, clean glass using newspaper instead of paper towel. sol2: Spray window cleaner on glass. For the best streak-free results, clean glass using construction paper instead of paper towel.',
            'targets':
                'Spray window cleaner on glass. For the best streak-free results, clean glass using newspaper instead of paper towel.'
        },
        {
            'inputs':
                'piqa goal: How do you prepare paint to repaint a house? sol1: You have to mix it very well. sol2: Just start painting right out of the can.',
            'targets':
                'You have to mix it very well.'
        },
    ])

  def test_preprocess_multiple_choice_winogrande(self):
    original_dataset = tfds.load(
        'winogrande:1.1.0', split='validation', shuffle_files=False).take(5)
    dataset = spot_preprocessors.preprocess_multiple_choice(
        original_dataset,
        choice_keys=['option1', 'option2'],
        question_key='sentence',
        task_name='winogrande',
    )
    assert_dataset(dataset, [
        {
            'inputs':
                'winogrande sentence: The woman avoided the hole but easily stepped over the pit, because the _ was very shallow. option1: hole option2: pit',
            'targets':
                'pit'
        },
        {
            'inputs':
                "winogrande sentence: So _ ignores Google to search for information because Betty trusts in it and Cynthia doesn't. option1: Betty option2: Cynthia",
            'targets':
                'Cynthia'
        },
        {
            'inputs':
                'winogrande sentence: Angela was a homebody while Amy loved to travel the world whenever they could. _ took a vacation to the beach over the summer. option1: Angela option2: Amy',
            'targets':
                'Amy'
        },
        {
            'inputs':
                'winogrande sentence: The hiking group had more food than water, so they tried to conserve the _ . option1: water option2: food',
            'targets':
                'water'
        },
        {
            'inputs':
                'winogrande sentence: Emily has never struggled with blood clots like Victoria has, because _ lives a sedentary, gluttonous lifestyle. option1: Emily option2: Victoria',
            'targets':
                'Victoria'
        },
    ])


if __name__ == '__main__':
  absltest.main()

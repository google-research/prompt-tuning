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

"""Sentiment datasets."""

import functools

from prompt_tuning.spot import preprocessors as spot_preprocessors
import seqio
from t5.data import preprocessors as t5_preprocessors
from t5.data import tasks as t5_tasks
from t5.evaluation import metrics as t5_metrics

TaskRegistry = seqio.TaskRegistry
MixtureRegistry = seqio.MixtureRegistry

# ============================= TFDS datasets ==================================
DATASETS = {
    'goemotions': {
        'tfds_name': 'goemotions:0.1.0',
        'preprocessor': spot_preprocessors.preprocess_goemotions,
    },
    'imdb_reviews': {
        'tfds_name':
            'imdb_reviews/plain_text:1.0.0',
        'preprocessor':
            functools.partial(
                spot_preprocessors.preprocess_text_classification,
                text_a_key='text',
                task_name='imdb_reviews',
                label_names=['negative', 'positive']),
    },
    'sentiment140': {
        'tfds_name':
            'sentiment140:1.0.0',
        'preprocessor':
            functools.partial(
                spot_preprocessors.preprocess_sentiment140,
                task_name='sentiment140',
                label_names=['negative', 'neutral', 'positive']),
    },
    'yelp_polarity_reviews': {
        'tfds_name':
            'yelp_polarity_reviews:0.2.0',
        'preprocessor':
            functools.partial(
                spot_preprocessors.preprocess_text_classification,
                text_a_key='text',
                task_name='yelp_polarity_reviews',
                label_names=['negative', 'positive']),
    },
}

# Register datasets
for dataset in DATASETS:
  version = f"v{DATASETS[dataset]['tfds_name'].split(':')[-1].replace('.', '')}"
  TaskRegistry.add(
      f'spot_{dataset.lower()}_{version}',
      source=seqio.TfdsDataSource(tfds_name=DATASETS[dataset]['tfds_name']),
      preprocessors=[
          DATASETS[dataset]['preprocessor'],
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      metric_fns=[t5_metrics.accuracy],
      output_features=t5_tasks.DEFAULT_OUTPUT_FEATURES)

# ============================ Others datasets =================================
CR_TRAIN_PATH = '/path/to/cr/train/file'
CR_EVAL_PATH = '/path/to/cr/eval/file'

# Register tasks
TaskRegistry.add(
    'spot_cr_v100',
    source=seqio.TextLineDataSource(
        split_to_filepattern={
            'train': CR_TRAIN_PATH,
            'validation': CR_EVAL_PATH
        },
        skip_header_lines=1),
    preprocessors=[
        functools.partial(
            t5_preprocessors.parse_tsv,
            field_names=['index', 'text', 'label']),
        functools.partial(
            spot_preprocessors.preprocess_text_classification,
            text_a_key='text',
            task_name='cr',
            label_names=['negative', 'positive']),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[t5_metrics.accuracy],
    output_features=t5_tasks.DEFAULT_OUTPUT_FEATURES)


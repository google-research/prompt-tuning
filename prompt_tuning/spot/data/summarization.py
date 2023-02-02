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

"""Summarization datasets."""

import functools

from prompt_tuning.data import preprocessors as pt_preprocessors
import seqio
from t5.data import tasks as t5_tasks
from t5.evaluation import metrics as t5_metrics

TaskRegistry = seqio.TaskRegistry
MixtureRegistry = seqio.MixtureRegistry

DATASETS = {
    'aeslc': {
        'tfds_name': 'aeslc:1.0.0',
        'source_key': 'email_body',
        'target_key': 'subject_line',
    },
    'billsum': {
        'tfds_name': 'billsum:3.0.0',
        'source_key': 'text',
        'target_key': 'summary',
    },
    'gigaword': {
        'tfds_name': 'gigaword:1.2.0',
        'source_key': 'document',
        'target_key': 'summary',
    },
    'cnn_dailymail': {
        'tfds_name': 'cnn_dailymail:3.2.0',
        'source_key': 'article',
        'target_key': 'highlights',
    },
    'multi_news': {
        'tfds_name': 'multi_news:1.0.0',
        'source_key': 'document',
        'target_key': 'summary',
    },
    'samsum': {
        'tfds_name': 'samsum:1.0.0',
        'source_key': 'dialogue',
        'target_key': 'summary',
    },
    'newsroom': {
        'tfds_name': 'newsroom:1.0.0',
        'source_key': 'text',
        'target_key': 'summary',
    },
}

# Register datasets
for dataset in DATASETS:
  version = f"v{DATASETS[dataset]['tfds_name'].split(':')[-1].replace('.', '')}"
  TaskRegistry.add(
      f'spot_{dataset.lower()}_{version}',
      source=seqio.TfdsDataSource(tfds_name=DATASETS[dataset]['tfds_name']),
      preprocessors=[
          functools.partial(
              pt_preprocessors.preprocess_text_generation,
              source_key=DATASETS[dataset]['source_key'],
              target_key=DATASETS[dataset]['target_key'],
              task_name=dataset.lower(),
              prefix='summarize:',
          ),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      metric_fns=[t5_metrics.rouge],
      output_features=t5_tasks.DEFAULT_OUTPUT_FEATURES)

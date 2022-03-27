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

from prompt_tuning.data import preprocessors as pt_preprocessors
from prompt_tuning.spot import preprocessors as spot_preprocessors
import seqio
from t5.data import tasks as t5_tasks
from t5.evaluation import metrics as t5_metrics

TaskRegistry = seqio.TaskRegistry
MixtureRegistry = seqio.MixtureRegistry

# ============================= TFDS datasets ==================================
DATASETS = {
    'common_gen': {
        'tfds_name':
            'gem/common_gen:1.1.0',
        'preprocessor':
            functools.partial(
                pt_preprocessors.preprocess_text_generation,
                source_key='concepts',
                target_key='target',
                task_name='common_gen',
                source_segment=' \\\\n ',
            ),
    },
    'dart': {
        'tfds_name':
            'gem/dart:1.1.0',
        'preprocessor':
            functools.partial(
                pt_preprocessors.preprocess_text_generation,
                source_key='tripleset',
                target_key='target',
                task_name='dart',
                source_segment=' \\\\n ',
            ),
    },
    'e2e_nlg': {
        'tfds_name': 'e2e_nlg:1.1.0',
        'preprocessor':
            functools.partial(
                pt_preprocessors.preprocess_text_generation,
                source_key='meaning_representation',
                target_key='target',
                task_name='e2e_nlg',
            ),
    },
    'schema_guided_dialog': {
        'tfds_name': 'schema_guided_dialog:1.1.0',
        'preprocessor':
            functools.partial(
                spot_preprocessors.preprocess_gem_sgd,
                task_name='schema_guided_dialog',
            ),
    },
    'web_nlg_en': {
        'tfds_name': 'web_nlg_en:1.1.0',
        'preprocessor':
            functools.partial(
                pt_preprocessors.preprocess_text_generation,
                source_key='input',
                target_key='target',
                task_name='web_nlg_en',
                source_segment=' \\\\n ',
            ),
    },
    'wiki_auto_asset_turk': {
        'tfds_name': 'wiki_auto_asset_turk:1.1.0',
        'preprocessor':
            functools.partial(
                pt_preprocessors.preprocess_text_generation,
                source_key='input',
                target_key='target',
                task_name='wiki_auto_asset_turk',
                prefix='simplify:',
            ),
    },
    'xsum': {
        'tfds_name': 'xsum:1.1.0',
        'preprocessor':
            functools.partial(
                pt_preprocessors.preprocess_text_generation,
                source_key='document',
                target_key='target',
                task_name='xsum',
                prefix='summarize:',
            ),
    },
    'wiki_lingua_english_en': {
        'tfds_name': 'wiki_lingua_english_en:1.1.0',
        'preprocessor':
            functools.partial(
                pt_preprocessors.preprocess_text_generation,
                source_key='source_aligned',
                target_key='target_aligned',
                task_name='wiki_lingua_english_en',
                prefix='summarize:',
                source_nested_key='en',
                target_nested_key='en',
            ),
    },
}

# Register datasets
for dataset in DATASETS:
  version = f"v{DATASETS[dataset]['tfds_name'].split(':')[-1].replace('.', '')}"
  TaskRegistry.add(
      f'spot_gem_{dataset.lower()}_{version}',
      source=seqio.TfdsDataSource(tfds_name=DATASETS[dataset]['tfds_name']),
      preprocessors=[
          DATASETS[dataset]['preprocessor'],
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      metric_fns=[t5_metrics.rouge, t5_metrics.bleu],
      output_features=t5_tasks.DEFAULT_OUTPUT_FEATURES)

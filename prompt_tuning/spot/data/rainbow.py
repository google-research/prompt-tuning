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

"""RAINBOW datasets."""

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
    'cosmos_qa': {
        'tfds_name':
            'cosmos_qa:1.0.0',
        'preprocessor':
            functools.partial(
                spot_preprocessors.preprocess_multiple_choice,
                choice_keys=['answer0', 'answer1', 'answer2', 'answer3'],
                context_key='context',
                question_key='question',
                task_name='cosmo_qa',
            ),
        'splits': ['train', 'validation', 'test'],
    },
    'hellaswag': {
        'tfds_name':
            'hellaswag:1.1.0',
        'preprocessor':
            functools.partial(
                spot_preprocessors.preprocess_multiple_choice,
                choice_keys=['endings', 'endings', 'endings', 'endings'],
                choice_nested_keys=[0, 1, 2, 3],
                question_key='context',
                task_name='hellaswag',
            ),
        'splits': ['train', 'validation', 'test'],
    },
    'piqa': {
        'tfds_name':
            'piqa:1.0.0',
        'preprocessor':
            functools.partial(
                spot_preprocessors.preprocess_multiple_choice,
                choice_keys=['sol1', 'sol2'],
                question_key='goal',
                task_name='piqa',
            ),
        'splits': ['train', 'validation'],
    },
    'winogrande': {
        'tfds_name':
            'winogrande:1.1.0',
        'preprocessor':
            functools.partial(
                spot_preprocessors.preprocess_multiple_choice,
                choice_keys=['option1', 'option2'],
                question_key='sentence',
                task_name='winogrande',
            ),
        'splits': ['train_xl', 'validation', 'test'],
    },
}

# Register datasets
for dataset in DATASETS:
  version = f"v{DATASETS[dataset]['tfds_name'].split(':')[-1].replace('.', '')}"
  TaskRegistry.add(
      f'spot_rainbow_{dataset.lower()}_{version}',
      source=seqio.TfdsDataSource(tfds_name=DATASETS[dataset]['tfds_name']),
      preprocessors=[
          functools.partial(
              spot_preprocessors.preprocess_text_classification,
              text_a_key=DATASETS[dataset]['text_a_key'],
              text_b_key=DATASETS[dataset]['text_b_key'],
              task_name=dataset,
              label_names=DATASETS[dataset]['label_names']),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      metric_fns=[t5_metrics.accuracy],
      output_features=t5_tasks.DEFAULT_OUTPUT_FEATURES)

# ============================ Others datasets =================================
ANLI_TRAIN_PATH = '/path/to/anli/train/file'
ANLI_EVAL_PATH = '/path/to/anli/eval/file'

# Register tasks
for dataset in DATASETS:
  TaskRegistry.add(
      'rom_rainbow_anli_v100',
      source=seqio.TextLineDataSource(
          split_to_filepattern={
              'train': ANLI_TRAIN_PATH,
              'validation': ANLI_EVAL_PATH,
          },
          skip_header_lines=1),
      preprocessors=[
          functools.partial(
              t5_preprocessors.parse_tsv,
              field_names=['index', 'obs1', 'obs2', 'hyp1', 'hyp2', 'label'],
          ),
          functools.partial(
              spot_preprocessors.preprocess_multiple_choice,
              choice_keys=['obs1', 'obs2', 'hyp1', 'hyp2'],
              task_name='abductive_nli',
          ),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      metric_fns=[t5_metrics.accuracy],
      output_features=t5_tasks.DEFAULT_OUTPUT_FEATURES)

SOCIALIQA_TRAIN_PATH = '/path/to/socialiqa/train/file'
SOCIALIQA_EVAL_PATH = '/path/to/socialiqa/eval/file'

# Register tasks
for dataset in DATASETS:
  TaskRegistry.add(
      'rom_rainbow_socialiqa_v100',
      source=seqio.TextLineDataSource(
          split_to_filepattern={
              'train': SOCIALIQA_TRAIN_PATH,
              'validation': SOCIALIQA_EVAL_PATH,
          },
          skip_header_lines=1),
      preprocessors=[
          functools.partial(
              t5_preprocessors.parse_tsv,
              field_names=[
                  'index', 'context', 'question', 'answerA', 'answerB',
                  'answerC', 'label'
              ],
          ),
          functools.partial(
              spot_preprocessors.preprocess_multiple_choice,
              choice_keys=['answerA', 'answerB', 'answerC'],
              context_key='context',
              question_key='question',
              task_name='socialiqa',
          ),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      metric_fns=[t5_metrics.accuracy],
      output_features=t5_tasks.DEFAULT_OUTPUT_FEATURES)


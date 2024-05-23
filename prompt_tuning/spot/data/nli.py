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

"""Summarization datasets."""

import functools

from prompt_tuning.spot.data import preprocessors as spot_preprocessors
import seqio
from t5.data import tasks as t5_tasks
from t5.evaluation import metrics as t5_metrics

TaskRegistry = seqio.TaskRegistry
MixtureRegistry = seqio.MixtureRegistry

DATASETS = {
    'anli_r1': {
        'tfds_name': 'anli/r1:0.1.0',
        'text_a_key': 'hypothesis',
        'text_b_key': 'context',
        'label_names': ['entailment', 'neutral', 'contradiction'],
    },
    'anli_r2': {
        'tfds_name': 'anli/r2:0.1.0',
        'text_a_key': 'hypothesis',
        'text_b_key': 'context',
        'label_names': ['entailment', 'neutral', 'contradiction'],
    },
    'anli_r3': {
        'tfds_name': 'anli/r3:0.1.0',
        'text_a_key': 'hypothesis',
        'text_b_key': 'context',
        'label_names': ['entailment', 'neutral', 'contradiction'],
    },
    'doc_nli': {
        'tfds_name': 'doc_nli:1.0.0',
        'text_a_key': 'hypothesis',
        'text_b_key': 'premise',
        'label_names': ['not_entailment', 'entailment'],
    },
    'snli': {
        'tfds_name': 'snli:1.1.0',
        'text_a_key': 'hypothesis',
        'text_b_key': 'premise',
        'label_names': ['entailment', 'neutral', 'contradiction'],
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

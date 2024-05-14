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

"""MRQA datasets.

Note that we do not include held-out out-of-domain datasets here.
"""

import functools

from prompt_tuning.data import preprocessors as pt_preprocessors
import seqio
from t5.data import postprocessors as t5_postprocessors
from t5.data import tasks as t5_tasks
from t5.evaluation import metrics as t5_metrics

TaskRegistry = seqio.TaskRegistry
MixtureRegistry = seqio.MixtureRegistry

DATASETS = {
    'squad': {
        'tfds_name': 'mrqa/squad:1.0.0',
    },
    'news_qa': {
        'tfds_name': 'mrqa/news_qa:1.0.0',
    },
    'trivia_qa': {
        'tfds_name': 'mrqa/trivia_qa:1.0.0',
    },
    'search_qa': {
        'tfds_name': 'mrqa/search_qa:1.0.0',
    },
    'hotpot_qa': {
        'tfds_name': 'mrqa/hotpot_qa:1.0.0',
    },
    'natural_questions': {
        'tfds_name': 'mrqa/natural_questions:1.0.0',
    },
}

# Register datasets
for dataset in DATASETS:
  version = f"v{DATASETS[dataset]['tfds_name'].split(':')[-1].replace('.', '')}"
  TaskRegistry.add(
      f'spot_mrqa_{dataset.lower()}_{version}',
      source=seqio.TfdsDataSource(tfds_name=DATASETS[dataset]['tfds_name']),
      preprocessors=[
          functools.partial(
              pt_preprocessors.mrqa,
              task_name=dataset.lower(),
          ),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=t5_postprocessors.qa,
      metric_fns=[t5_metrics.squad],
      output_features=t5_tasks.DEFAULT_OUTPUT_FEATURES)

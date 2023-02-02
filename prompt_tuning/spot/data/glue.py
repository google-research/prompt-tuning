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

"""GLUE datasets.

Register GLUE datasets using their latest TFDS versions.
"""

import seqio
from t5.data import tasks as t5_tasks
from t5.data.glue_utils import get_glue_metric
from t5.data.glue_utils import get_glue_postprocess_fn
from t5.data.glue_utils import get_glue_text_preprocessor
import tensorflow_datasets as tfds

TaskRegistry = seqio.TaskRegistry
MixtureRegistry = seqio.MixtureRegistry

for b in tfds.text.glue.Glue.builder_configs.values():
  if b.name != 'ax' and not b.name.startswith('mnli'):
    TaskRegistry.add(
        f'spot_glue_{b.name}_v200',
        source=seqio.TfdsDataSource(tfds_name=f'glue/{b.name}:2.0.0'),
        preprocessors=[
            get_glue_text_preprocessor(b),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        metric_fns=get_glue_metric(b.name),
        output_features=t5_tasks.DEFAULT_OUTPUT_FEATURES,
        postprocess_fn=get_glue_postprocess_fn(b))
  elif b.name == 'mnli':
    # We will register MNLI's validation sets later
    TaskRegistry.add(
        f'spot_glue_{b.name}_train_v200',
        source=seqio.TfdsDataSource(
            tfds_name=f'glue/{b.name}:2.0.0', splits=['train']),
        preprocessors=[
            get_glue_text_preprocessor(b),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        metric_fns=get_glue_metric(b.name),
        output_features=t5_tasks.DEFAULT_OUTPUT_FEATURES,
        postprocess_fn=get_glue_postprocess_fn(b))

# Register MNLI's validation sets so we can see the results on Tensorboard
b = tfds.text.glue.Glue.builder_configs['mnli']
for split in ['validation_matched', 'validation_mismatched']:
  TaskRegistry.add(
      f'spot_glue_mnli_{split}_v200',
      source=seqio.TfdsDataSource(
          tfds_name='glue/mnli:2.0.0', splits={'validation': split}),
      preprocessors=[
          get_glue_text_preprocessor(b),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      metric_fns=get_glue_metric(b.name),
      output_features=t5_tasks.DEFAULT_OUTPUT_FEATURES,
      postprocess_fn=get_glue_postprocess_fn(b))

# Create MNLI mixture
MixtureRegistry.add(
    'spot_glue_mnli_and_dev_v200', ([
        'spot_glue_mnli_train_v200',
        'spot_glue_mnli_validation_matched_v200',
        'spot_glue_mnli_validation_mismatched_v200',
    ]),
    default_rate=1.0)

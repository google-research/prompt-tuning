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

"""Similarity datasets."""

import functools

from prompt_tuning.spot import preprocessors as spot_preprocessors
import seqio
from t5.data import postprocessors as t5_postprocessors
from t5.data import preprocessors as t5_preprocessors
from t5.data import tasks as t5_tasks
from t5.evaluation import metrics as t5_metrics

TaskRegistry = seqio.TaskRegistry
MixtureRegistry = seqio.MixtureRegistry

# ============================ CxC =================================
CXC_TRAIN_PATH = '/path/to/cxc/train/file'
CXC_EVAL_PATH = '/path/to/cxc/eval/file'

# Register tasks
TaskRegistry.add(
    'spot_cxc_v100',
    source=seqio.TextLineDataSource(
        split_to_filepattern={
            'train': CXC_TRAIN_PATH,
            'validation': CXC_EVAL_PATH
        },
        skip_header_lines=1),
    preprocessors=[
        functools.partial(
            t5_preprocessors.parse_tsv,
            field_names=['index', 'sentence1', 'sentence2', 'label']),
        functools.partial(
            spot_preprocessors.preprocess_text_similarity,
            text_a_key='sentence1',
            text_b_key='sentence2',
            task_name='cxc',
            id_key='index'),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=t5_postprocessors.string_to_float,
    metric_fns=[t5_metrics.accuracy],
    output_features=t5_tasks.DEFAULT_OUTPUT_FEATURES)

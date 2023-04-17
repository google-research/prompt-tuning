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

"""Tasks that use the c4 dataset."""

import functools
from prompt_tuning.data import features
from prompt_tuning.data import preprocessors
import seqio
from t5.data import preprocessors as t5_preprocessors



# ===== BART (Lewis et. al., 2019) Pre-training-like objectives =====
seqio.TaskRegistry.add(
    "c4_v220_bart_text_infilling",
    source=seqio.TfdsDataSource(tfds_name="c4/en:2.2.0"),
    preprocessors=[
        functools.partial(
            t5_preprocessors.rekey,
            key_map={
                "inputs": None,
                "targets": "text",
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.text_infilling,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=features.T5_FEATURES,
    metric_fns=[],
    postprocess_fn=None
)


seqio.TaskRegistry.add(
    "c4_v220_bart_token_deletion",
    source=seqio.TfdsDataSource(tfds_name="c4/en:2.2.0"),
    preprocessors=[
        functools.partial(
            t5_preprocessors.rekey,
            key_map={
                "inputs": None,
                "targets": "text",
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.token_deletion,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=features.T5_FEATURES,
    metric_fns=[],
    postprocess_fn=None
)

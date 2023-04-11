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

"""Register SST2 Generative and Rank Classification Eval mixture."""

import functools
from prompt_tuning.data import features
from prompt_tuning.data import preprocessors as pt_preprocessors
from prompt_tuning.recycling.data import rank_classification as pr_rc
import seqio
from t5.data import glue_utils
from t5.data import postprocessors as t5_postprocessors
from t5.data import preprocessors as t5_preprocessors
from t5.evaluation import metrics as t5_metrics
import tensorflow_datasets as tfds


builder = tfds.text.glue.Glue.builder_configs["sst2"]

# Training and Dev set split.
seqio.TaskRegistry.add(
    "pr:glue_sst2_v200_train",
    source=seqio.TfdsDataSource(
        tfds_name="glue/sst2:2.0.0",
        splits={
            "train": "train[:-1000]",
            "validation": "train[-1000:]"
        }
    ),
    preprocessors=[
        glue_utils.get_glue_text_preprocessor(builder),
        pt_preprocessors.remove_first_text_token,
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=glue_utils.get_glue_postprocess_fn(builder),
    metric_fns=glue_utils.get_glue_metric(builder.name),
    output_features=features.T5_FEATURES
)

# Generative Test split
seqio.TaskRegistry.add(
    "pr:glue_sst2_v200_test",
    source=seqio.TfdsDataSource(
        tfds_name="glue/sst2:2.0.0",
        splits=("validation",)
    ),
    preprocessors=[
        glue_utils.get_glue_text_preprocessor(builder),
        pt_preprocessors.remove_first_text_token,
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=glue_utils.get_glue_postprocess_fn(builder),
    metric_fns=glue_utils.get_glue_metric(builder.name),
    output_features=features.T5_FEATURES
)

# Rank classification Test split
seqio.TaskRegistry.add(
    "pr:glue_sst2_v200_test_rc",
    source=seqio.TfdsDataSource(
        tfds_name="glue/sst2:2.0.0",
        splits=("validation",)
    ),
    preprocessors=[
        glue_utils.get_glue_text_preprocessor(builder),
        pt_preprocessors.remove_first_text_token,
        functools.partial(
            t5_preprocessors.rank_classification,
            inputs_fn=functools.partial(pr_rc.get_inputs,
                                        labels=builder.label_classes),
            targets_fn=functools.partial(pr_rc.get_targets,
                                         labels=builder.label_classes),
            is_correct_fn=functools.partial(pr_rc.get_correct,
                                            labels=builder.label_classes)
        ),
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=t5_postprocessors.rank_classification,
    metric_fns=[functools.partial(
        t5_metrics.rank_classification,
        num_classes=2)],
    output_features=features.T5_FEATURES
)

# Mix training and eval settings together for training and early stopping.
seqio.MixtureRegistry.add(
    "pr:glue_sst2_v200",
    ["pr:glue_sst2_v200_train",
     "pr:glue_sst2_v200_test",
     "pr:glue_sst2_v200_test_rc"],
    default_rate=1.0)

# Mix together to run generative and RC evals at onces from eval.py
seqio.MixtureRegistry.add(
    "pr:glue_sst2_v200_eval",
    ["pr:glue_sst2_v200_test_rc",
     "pr:glue_sst2_v200_test"],
    default_rate=1.0)

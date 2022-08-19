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

"""Register QQP Training and Eval mixtures.

We do this in it's own file for now because we want to explore verbalizers.
"""

import functools
from typing import Sequence
from prompt_tuning.data import features
from prompt_tuning.data import preprocessors as pt_preprocessors
from prompt_tuning.recycling.data import rank_classification as pr_rc
import seqio
from t5.data import glue_utils
from t5.data import postprocessors as t5_postprocessors
from t5.data import preprocessors as t5_preprocessors
from t5.evaluation import metrics as t5_metrics
import tensorflow_datasets as tfds


b = tfds.text.glue.Glue.builder_configs["qqp"]

# This is similar to what gets called in
# `t5.data.glue_utils.get_glue_text_preprocessor` but we can change the
# `label_names` later.
qqp_text_preprocessor = functools.partial(
    t5_preprocessors.glue,
    benchmark_name="qqp",
    feature_names=None)

qqp_postprocess_fn = t5_postprocessors.string_label_to_class_id


def verbs_as_str(verbs: Sequence[str]) -> str:
  """Stringify the verbalizers to add to the task name."""
  return "_".join("_spc_".join(v.split()) for v in verbs)


VERBALIZERS = (
    list(b.label_classes),  # [duplicate, not_duplicate]
    list(map(str, [0, 1])),
    ["unique", "duplicate"],
    ["not duplicate", "duplicate"],
)


for verbalizers in VERBALIZERS:
  # Training task
  seqio.TaskRegistry.add(
      f"pr:glue_qqp_v200_{verbs_as_str(verbalizers)}_verbalizers_training",
      source=seqio.TfdsDataSource(
          tfds_name="glue/qqp:2.0.0",
          splits={
              "train": "train[:-10000]",
              "validation": "train[-10000:]"
          },
      ),
      preprocessors=[
          functools.partial(
              qqp_text_preprocessor,
              label_names=verbalizers),
          pt_preprocessors.remove_first_text_token,
          seqio.preprocessors.tokenize,
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=functools.partial(
          qqp_postprocess_fn,
          label_classes=verbalizers),
      metric_fns=glue_utils.get_glue_metric("qqp"),
      output_features=features.T5_FEATURES)
  # Training Eval task
  seqio.TaskRegistry.add(
      f"pr:glue_qqp_v200_{verbs_as_str(verbalizers)}_verbalizers_test",
      source=seqio.TfdsDataSource(
          tfds_name="glue/qqp:2.0.0",
          splits=("validation",),
      ),
      preprocessors=[
          functools.partial(
              qqp_text_preprocessor,
              label_names=verbalizers),
          pt_preprocessors.remove_first_text_token,
          seqio.preprocessors.tokenize,
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=functools.partial(
          qqp_postprocess_fn,
          label_classes=verbalizers),
      metric_fns=glue_utils.get_glue_metric("qqp"),
      output_features=features.T5_FEATURES)
  # Rank Classification Eval Task
  seqio.TaskRegistry.add(
      f"pr:glue_qqp_v200_{verbs_as_str(verbalizers)}_verbalizers_rc",
      source=seqio.TfdsDataSource(
          tfds_name="glue/qqp:2.0.0",
          splits=("validation",),
      ),
      preprocessors=[
          functools.partial(
              qqp_text_preprocessor,
              label_names=verbalizers),
          pt_preprocessors.remove_first_text_token,
          functools.partial(
              t5_preprocessors.rank_classification,
              inputs_fn=functools.partial(
                  pr_rc.get_inputs,
                  labels=verbalizers),
              targets_fn=functools.partial(
                  pr_rc.get_targets,
                  labels=verbalizers),
              is_correct_fn=functools.partial(
                  pr_rc.get_correct,
                  labels=verbalizers),
          ),
          seqio.preprocessors.tokenize,
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=t5_postprocessors.rank_classification,
      metric_fns=[functools.partial(
          t5_metrics.rank_classification,
          num_classes=len(verbalizers),
          normalize_by_target_length=False
      )],
      output_features=features.T5_FEATURES
  )
  seqio.TaskRegistry.add(
      f"pr:glue_qqp_v200_{verbs_as_str(verbalizers)}_verbalizers_rc_normed",
      source=seqio.TfdsDataSource(
          tfds_name="glue/qqp:2.0.0",
          splits=("validation",),
      ),
      preprocessors=[
          functools.partial(
              qqp_text_preprocessor,
              label_names=verbalizers),
          pt_preprocessors.remove_first_text_token,
          functools.partial(
              t5_preprocessors.rank_classification,
              inputs_fn=functools.partial(
                  pr_rc.get_inputs,
                  labels=verbalizers),
              targets_fn=functools.partial(
                  pr_rc.get_targets,
                  labels=verbalizers),
              is_correct_fn=functools.partial(
                  pr_rc.get_correct,
                  labels=verbalizers),
          ),
          seqio.preprocessors.tokenize,
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=t5_postprocessors.rank_classification,
      metric_fns=[functools.partial(
          t5_metrics.rank_classification,
          num_classes=len(verbalizers),
          normalize_by_target_length=True
      )],
      output_features=features.T5_FEATURES
  )
  # Mix them together for easy use.
  seqio.MixtureRegistry.add(
      f"pr:glue_qqp_v200_{verbs_as_str(verbalizers)}_verbalizers",
      [
          f"pr:glue_qqp_v200_{verbs_as_str(verbalizers)}_verbalizers_rc",
          f"pr:glue_qqp_v200_{verbs_as_str(verbalizers)}_verbalizers_rc_normed",
          f"pr:glue_qqp_v200_{verbs_as_str(verbalizers)}_verbalizers_training",
          f"pr:glue_qqp_v200_{verbs_as_str(verbalizers)}_verbalizers_test",
      ],
      default_rate=1.0
  )
  seqio.MixtureRegistry.add(
      f"pr:glue_qqp_v200_{verbs_as_str(verbalizers)}_verbalizers_eval",
      [
          f"pr:glue_qqp_v200_{verbs_as_str(verbalizers)}_verbalizers_rc",
          f"pr:glue_qqp_v200_{verbs_as_str(verbalizers)}_verbalizers_rc_normed",
          f"pr:glue_qqp_v200_{verbs_as_str(verbalizers)}_verbalizers_test",
      ],
      default_rate=1.0
  )

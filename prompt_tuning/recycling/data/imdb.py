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

"""Register IMDB Generative and Rank Classification Eval mixtures."""

import functools
from typing import Sequence
from prompt_tuning.data import features
from prompt_tuning.recycling.data import metrics as pr_metrics
from prompt_tuning.recycling.data import preprocessors as pr_preprocessors
from prompt_tuning.recycling.data import rank_classification as pr_rc
import seqio
from t5.data import postprocessors as t5_postprocessors
from t5.data import preprocessors as t5_preprocessors
from t5.evaluation import metrics as t5_metrics


# Are we good at recycling sentiment datasets or at manipulating the `positive`/
# `negative` labels.
VERBALIZERS = (
    list(map(str, [0, 1])),
    ("negative", "positive")
)


def verbs_as_str(verbs: Sequence[str]) -> str:
  """Stringify the verbalizers to add them to the task name."""
  return "_".join(verbs)


rekey = functools.partial(
    seqio.preprocessors.rekey,
    key_map={
        "inputs": "text",
        "targets": "label",
    })


for verbalizers in VERBALIZERS:
  # Training and Dev set split.
  idx_to_str = functools.partial(
      pr_preprocessors.label_idx_to_string,
      label_names=verbalizers)
  postprocess_fn = functools.partial(
      t5_postprocessors.string_label_to_class_id,
      label_classes=verbalizers)

  seqio.TaskRegistry.add(
      f"pr:imdb_v100_{verbs_as_str(verbalizers)}_verbalizers_train",
      source=seqio.TfdsDataSource(
          tfds_name="imdb_reviews/plain_text:1.0.0",
          splits={
              "train": "train[:-1000]",
              "validation": "train[-1000:]"
          }
      ),
      preprocessors=[
          idx_to_str,
          rekey,
          seqio.preprocessors.tokenize,
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=postprocess_fn,
      metric_fns=[t5_metrics.accuracy],
      output_features=features.T5_FEATURES
  )

  # Generative Test split
  seqio.TaskRegistry.add(
      f"pr:imdb_v100_{verbs_as_str(verbalizers)}_verbalizers_test",
      source=seqio.TfdsDataSource(
          tfds_name="imdb_reviews/plain_text:1.0.0",
          splits={"validation": "test"}
      ),
      preprocessors=[
          idx_to_str,
          rekey,
          seqio.preprocessors.tokenize,
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=postprocess_fn,
      metric_fns=[t5_metrics.accuracy],
      output_features=features.T5_FEATURES
  )

  if verbalizers == ["0", "1"]:
    normed_rc = [
        functools.partial(
            pr_metrics.prefix_metric_names,
            metric_fn=functools.partial(
                t5_metrics.rank_classification,
                num_classes=len(verbalizers),
                normalize_by_target_length=True),
            metric_prefix="normed")
    ]
  else:
    normed_rc = []

  # Rank classification Test split
  seqio.TaskRegistry.add(
      f"pr:imdb_v100_{verbs_as_str(verbalizers)}_verbalizers_test_rc",
      source=seqio.TfdsDataSource(
          tfds_name="imdb_reviews/plain_text:1.0.0",
          splits={"validation": "test"}
      ),
      preprocessors=[
          idx_to_str,
          rekey,
          functools.partial(
              t5_preprocessors.rank_classification,
              inputs_fn=functools.partial(pr_rc.get_inputs,
                                          labels=verbalizers),
              targets_fn=functools.partial(pr_rc.get_targets,
                                           labels=verbalizers),
              is_correct_fn=functools.partial(pr_rc.get_correct,
                                              labels=verbalizers)
          ),
          seqio.preprocessors.tokenize,
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=t5_postprocessors.rank_classification,
      metric_fns=[
          functools.partial(
              t5_metrics.rank_classification,
              num_classes=len(verbalizers))
      ] + normed_rc,
      output_features=features.T5_FEATURES
  )

  # Mix training and eval settings together for training and early stopping.
  seqio.MixtureRegistry.add(
      f"pr:imdb_v100_{verbs_as_str(verbalizers)}_verbalizers",
      [f"pr:imdb_v100_{verbs_as_str(verbalizers)}_verbalizers_train",
       f"pr:imdb_v100_{verbs_as_str(verbalizers)}_verbalizers_test",
       f"pr:imdb_v100_{verbs_as_str(verbalizers)}_verbalizers_test_rc"],
      default_rate=1.0)

  # Mix together to run generative and RC evals at onces from eval.py
  seqio.MixtureRegistry.add(
      f"pr:imdb_v100_{verbs_as_str(verbalizers)}_verbalizers_eval",
      [f"pr:imdb_v100_{verbs_as_str(verbalizers)}_verbalizers_test",
       f"pr:imdb_v100_{verbs_as_str(verbalizers)}_verbalizers_test_rc"],
      default_rate=1.0)

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

"""QA and zeroshot domain transfer QA tasks and mixtures."""

import functools
from prompt_tuning.data import constants
from prompt_tuning.data import features
from prompt_tuning.data import metrics as pt_metrics
from prompt_tuning.data import postprocessors as pt_postprocessors
from prompt_tuning.data import preprocessors as pt_preprocessors
import seqio
from t5.data import postprocessors
from t5.data import preprocessors
from t5.evaluation import metrics


MRQA_OUT_OF_DOMAIN = ("bio_asq",
                      "drop",
                      "duo_rc",
                      "race",
                      "relation_extraction",
                      "textbook_qa")
MRQA_IN_DOMAIN = ("news_qa",
                  "trivia_qa",
                  "search_qa",
                  "hotpot_qa",
                  "natural_questions")

for model_prefix, feats in features.MODEL_TO_FEATURES.items():
  # This is the SQuAD but it can output examples to tensorboard.
  seqio.TaskRegistry.add(
      f"{model_prefix}squad_v010_allanswers_examples",
      source=seqio.TfdsDataSource(tfds_name="squad/v1.1:3.0.0"),
      preprocessors=[
          preprocessors.squad,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=functools.partial(
          pt_postprocessors.postprocess_with_examples,
          postprocessors.qa,
          example_fields=(
              constants.TARGET_TEXT,
              constants.CONTEXT_TEXT,
              constants.QUESTION_TEXT,
              constants.ANSWERS_TEXT,
          )),
      metric_fns=[
          functools.partial(pt_metrics.metric_with_examples, metrics.squad),
          functools.partial(
              pt_metrics.text_examples,
              task_name="squad",
              format_fn=pt_metrics.format_qa),
      ],
      output_features=feats)

  # ========== MRQA 2019 Shared-task on Generalization of QA ==========
  for mrqa_name in MRQA_OUT_OF_DOMAIN:
    seqio.TaskRegistry.add(
        f"{model_prefix}mrqa_{mrqa_name}_v100_examples",
        source=seqio.TfdsDataSource(
            tfds_name=f"mrqa/{mrqa_name}:1.0.0",
            splits={"validation": "test"}),
        preprocessors=[
            pt_preprocessors.mrqa,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        postprocess_fn=functools.partial(
            pt_postprocessors.postprocess_with_examples,
            postprocessors.qa,
            example_fields=(constants.TARGET_TEXT, constants.CONTEXT_TEXT,
                            constants.QUESTION_TEXT, constants.ANSWERS_TEXT)),
        metric_fns=[
            functools.partial(pt_metrics.metric_with_examples, metrics.squad),
            functools.partial(
                pt_metrics.text_examples,
                task_name=mrqa_name,
                format_fn=pt_metrics.format_qa)
        ],
        output_features=feats)

  for mrqa_name in MRQA_IN_DOMAIN:
    seqio.TaskRegistry.add(
        f"{model_prefix}mrqa_{mrqa_name}_v100_examples",
        source=seqio.TfdsDataSource(
            tfds_name=f"mrqa/{mrqa_name}:1.0.0"),
        preprocessors=[
            pt_preprocessors.mrqa,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        postprocess_fn=functools.partial(
            pt_postprocessors.postprocess_with_examples,
            postprocessors.qa,
            example_fields=(constants.TARGET_TEXT, constants.CONTEXT_TEXT,
                            constants.QUESTION_TEXT, constants.ANSWERS_TEXT)),
        metric_fns=[
            functools.partial(pt_metrics.metric_with_examples, metrics.squad),
            functools.partial(
                pt_metrics.text_examples,
                task_name=mrqa_name,
                format_fn=pt_metrics.format_qa)
        ],
        output_features=feats)

  # ========== Mixtures for Zero-Shot ==========
  # Multiple mixtures, each one is train on SQuAD and evaluation on one of the
  # MRQA out-of-domain datasets.
  for mrqa_name in MRQA_OUT_OF_DOMAIN:
    seqio.MixtureRegistry.add(
        f"{model_prefix}squad_to_{mrqa_name}_examples", [
            f"{model_prefix}squad_v010_allanswers_examples",
            f"{model_prefix}mrqa_{mrqa_name}_v100_examples"
        ],
        default_rate=1.0)

  # A single mixture that is train on SQuAD, evaluate on all of the MRQA
  # out-of-domain datasets.
  seqio.MixtureRegistry.add(
      f"{model_prefix}squad_to_mrqa_examples",
      [f"{model_prefix}squad_v010_allanswers_examples"] +
      [f"{model_prefix}mrqa_{name}_v100_examples"
       for name in MRQA_OUT_OF_DOMAIN],
      default_rate=1.0)
